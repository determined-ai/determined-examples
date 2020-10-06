"""
This example demonstrates how to train a Cycle GAN with Determined PyTorch API.

The PyTorch API supports multiple model graphs, optimizers, and LR
schedulers. Those objects should be created and wrapped in the trial class's
__init__ method. Then in train_batch(), you can run forward and backward passes
and step the optimizer according to your requirements.
"""
import itertools
from typing import Any, Dict, Union, Sequence

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import make_grid

from determined.pytorch import PyTorchTrial, PyTorchTrialContext, DataLoader, LRScheduler
from determined.tensorboard.metric_writers.pytorch import TorchWriter

from models import *
from datasets import *
from utils import *


cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]

class CycleGANTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        self.context = context
        self.logger = TorchWriter()

        self.dataset_path = f"""{self.context.get_data_config()["downloaded_path"]}/{self.context.get_data_config()["dataset_name"]}"""

        # Initialize the models.
        input_shape = (
            context.get_data_config()["channels"],
            context.get_data_config()["img_height"],
            context.get_data_config()["img_width"]
        )
        self.G_AB = self.context.wrap_model(GeneratorResNet(input_shape, context.get_hparam("n_residual_blocks")))
        self.G_BA = self.context.wrap_model(GeneratorResNet(input_shape, context.get_hparam("n_residual_blocks")))
        self.D_A = self.context.wrap_model(Discriminator(input_shape))
        self.D_B = self.context.wrap_model(Discriminator(input_shape))

        # Losses
        self.criterion_GAN = self.context.wrap_model(torch.nn.MSELoss())
        self.criterion_cycle = self.context.wrap_model(torch.nn.L1Loss())
        self.criterion_identity = self.context.wrap_model(torch.nn.L1Loss())

        # Initialize weights
        self.G_AB.apply(weights_init_normal)
        self.G_BA.apply(weights_init_normal)
        self.D_A.apply(weights_init_normal)
        self.D_B.apply(weights_init_normal)

        # Initialize the optimizers and learning rate scheduler.
        lr = context.get_hparam("lr")
        b1 = context.get_hparam("b1")
        b2 = context.get_hparam("b2")
        n_epochs = context.get_experiment_config()["searcher"]["max_length"]["epochs"]
        decay_epoch = context.get_hparam("decay_epoch")
        self.optimizer_G = self.context.wrap_optimizer(torch.optim.Adam(
            itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()), lr=lr, betas=(b1, b2)
        ))
        self.optimizer_D_A = self.context.wrap_optimizer(torch.optim.Adam(self.D_A.parameters(), lr=lr, betas=(b1, b2)))
        self.optimizer_D_B = self.context.wrap_optimizer(torch.optim.Adam(self.D_B.parameters(), lr=lr, betas=(b1, b2)))
        self.lr_scheduler_G = self.context.wrap_lr_scheduler(torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_G, lr_lambda=LambdaLR(n_epochs, 0, decay_epoch).step
        ), step_mode=LRScheduler.StepMode.STEP_EVERY_EPOCH)
        self.lr_scheduler_D_A = self.context.wrap_lr_scheduler(torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_D_A, lr_lambda=LambdaLR(n_epochs, 0, decay_epoch).step
        ), step_mode=LRScheduler.StepMode.STEP_EVERY_EPOCH)
        self.lr_scheduler_D_B = self.context.wrap_lr_scheduler(torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_D_B, lr_lambda=LambdaLR(n_epochs, 0, decay_epoch).step
        ), step_mode=LRScheduler.StepMode.STEP_EVERY_EPOCH)

        # Buffers of previously generated samples
        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        # Image transformations
        img_height = self.context.get_data_config()["img_height"]
        img_width = self.context.get_data_config()["img_width"]
        self.transforms_ = [
            transforms.Resize(int(img_height * 1.12), Image.BICUBIC),
            transforms.RandomCrop((img_height, img_width)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]

        # Test images that are used for displaying in Tensorboard.
        self.test_dataloader = torch.utils.data.DataLoader(
            ImageDataset(self.dataset_path, transforms_=self.transforms_, unaligned=True, mode="test"),
            batch_size=5,
            shuffle=True,
            num_workers=1,
        )

    def build_training_data_loader(self) -> DataLoader:
        return DataLoader(
            ImageDataset(self.dataset_path, transforms_=self.transforms_, unaligned=True),
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=True,
            num_workers=self.context.get_data_config()["n_cpu"],
        )

    def build_validation_data_loader(self) -> DataLoader:
        return DataLoader(
            ImageDataset(self.dataset_path, transforms_=self.transforms_, unaligned=True, mode="test"),
            batch_size=5,
            shuffle=True,
            num_workers=1,
        )

    def sample_images(self, prefix, batch_idx):
        imgs = next(iter(self.test_dataloader))
        self.G_AB.eval()
        self.G_BA.eval()
        real_A = Variable(imgs["A"].type(Tensor))
        fake_B = self.G_AB(real_A)
        real_B = Variable(imgs["B"].type(Tensor))
        fake_A = self.G_BA(real_B)
        # Arange images along x-axis
        real_A = make_grid(real_A, nrow=5, normalize=True)
        real_B = make_grid(real_B, nrow=5, normalize=True)
        fake_A = make_grid(fake_A, nrow=5, normalize=True)
        fake_B = make_grid(fake_B, nrow=5, normalize=True)
        # Arange images along y-axis
        image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
        self.logger.writer.add_images(f"{prefix}", image_grid, batch_idx, dataformats="CHW")

    def train_batch(
        self, batch: TorchData, epoch_idx: int, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        imgs, _ = batch

        # Set model input
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *self.D_A.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *self.D_A.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        self.G_AB.requires_grad_(True)
        self.G_BA.requires_grad_(True)
        self.D_A.requires_grad_(False)
        self.D_B.requires_grad_(False)

        self.G_AB.train()
        self.G_BA.train()

        # Identity loss
        loss_id_A = self.criterion_identity(self.G_BA(real_A), real_A)
        loss_id_B = self.criterion_identity(self.G_AB(real_B), real_B)

        loss_identity = (loss_id_A + loss_id_B) / 2

        # GAN loss
        fake_B = self.G_AB(real_A)
        loss_GAN_AB = self.criterion_GAN(self.D_B(fake_B), valid)
        fake_A = self.G_BA(real_B)
        loss_GAN_BA = self.criterion_GAN(self.D_A(fake_A), valid)

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle loss
        recov_A = self.G_BA(fake_B)
        loss_cycle_A = self.criterion_cycle(recov_A, real_A)
        recov_B = self.G_AB(fake_A)
        loss_cycle_B = self.criterion_cycle(recov_B, real_B)

        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Total loss
        lambda_cyc = self.context.get_hparam("lambda_cyc")
        lambda_id = self.context.get_hparam("lambda_id")
        loss_G = loss_GAN + lambda_cyc * loss_cycle + lambda_id * loss_identity

        self.context.backward(loss_G)
        self.optimizer_G.step()
        self.optimizer_G.zero_grad()

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        # Set `requires_grad_` to only update parameters on the discriminator A.
        self.G_AB.requires_grad_(False)
        self.G_BA.requires_grad_(False)
        self.D_A.requires_grad_(True)
        self.D_B.requires_grad_(False)

        # Real loss
        loss_real = self.criterion_GAN(self.D_A(real_A), valid)
        # Fake loss (on batch of previously generated samples)
        fake_A_ = self.fake_A_buffer.push_and_pop(fake_A)
        loss_fake = self.criterion_GAN(self.D_A(fake_A_.detach()), fake)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2

        self.context.backward(loss_D_A)
        self.optimizer_D_A.step()
        self.optimizer_D_A.zero_grad()

        # -----------------------
        #  Train Discriminator B
        # -----------------------

        # Set `requires_grad_` to only update parameters on the discriminator A.
        self.G_AB.requires_grad_(False)
        self.G_BA.requires_grad_(False)
        self.D_A.requires_grad_(False)
        self.D_B.requires_grad_(True)

        # Real loss
        loss_real = self.criterion_GAN(self.D_B(real_B), valid)
        # Fake loss (on batch of previously generated samples)
        fake_B_ = self.fake_B_buffer.push_and_pop(fake_B)
        loss_fake = self.criterion_GAN(self.D_B(fake_B_.detach()), fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2

        self.context.backward(loss_D_B)
        self.optimizer_D_B.step()
        self.optimizer_D_B.zero_grad()

        loss_D = (loss_D_A + loss_D_B) / 2

        # --------------
        #  Log Progress
        # --------------

        # If at sample interval save image
        global_records = self.context.get_global_batch_size() * batch_idx
        sample_interval = self.context.get_data_config()["sample_interval"]
        if self.context.distributed.get_rank() == 0 and global_records % sample_interval == 0:
            self.sample_images(self.context.get_data_config()["dataset_name"], batch_idx)

        return {
            "loss_D": loss_D,
            "loss_G": loss_G,
            "loss_GAN": loss_GAN,
            "loss_cycle": loss_cycle,
            "loss_identity": loss_identity,
        }

    def evaluate_batch(self, batch: TorchData) -> Dict[str, Any]:
        # Set model input
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))
        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *self.D_A.output_shape))), requires_grad=False)
        # Real loss
        loss_real_D_A = self.criterion_GAN(self.D_A(real_A), valid)
        loss_real_D_B = self.criterion_GAN(self.D_B(real_B), valid)
        # Total loss
        loss_real_D = (loss_real_D_A + loss_real_D_B) / 2

        return {
            "loss_real_D": loss_real_D,
            "loss_real_D_A": loss_real_D_A,
            "loss_real_D_B": loss_real_D_B,
        }
