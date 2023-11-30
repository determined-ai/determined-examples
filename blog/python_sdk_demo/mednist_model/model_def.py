import os
from typing import Any, Dict, Sequence, Union

import determined
from determined.pytorch import (
    DataLoader,
    LRScheduler,
    PyTorchTrial,
    PyTorchTrialContext,
)
import medmnist
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import wget

from net import Net

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]
DATASET_ROOT = "datasets"


class MyMEDMnistTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        self.context = context
        hparams = self.context.get_hparam

        self.info = medmnist.INFO[hparams("data_flag")]
        n_channels = self.info["n_channels"]
        n_classes = len(self.info["label"])
        self.task = self.info["task"]

        model = Net(n_channels, n_classes)
        self.model = self.context.wrap_model(model)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hparams("lr"),
            weight_decay=hparams("weight_decay"),
            betas=(hparams("beta1"), hparams("beta2")),
        )
        self.optimizer = self.context.wrap_optimizer(optimizer)
        self.criterion = nn.CrossEntropyLoss()

        num_epochs = self.context.get_experiment_config()["searcher"]["max_length"]["epochs"]
        milestones = [int(0.5 * num_epochs), int(0.75 * num_epochs)]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=milestones,
            gamma=hparams("gamma"),
        )
        self.lr_sch = self.context.wrap_lr_scheduler(
            scheduler, step_mode=LRScheduler.StepMode.STEP_EVERY_EPOCH
        )

        os.makedirs(DATASET_ROOT, exist_ok=True)

        data_url = context.get_data_config().get("url")
        if data_url:
            wget.download(
                data_url,
                out=os.path.join(DATASET_ROOT, f"{hparams('data_flag')}.npz"),
            )

    def build_training_data_loader(self) -> DataLoader:
        DataClass = getattr(medmnist, self.info["python_class"])
        data_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
        )

        train_dataset = DataClass(
            split="train",
            transform=data_transform,
            download=True,
            as_rgb=True,
            root=DATASET_ROOT,
        )
        train_loader = determined.pytorch.DataLoader(
            dataset=train_dataset,
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=True,
        )

        return train_loader

    def build_validation_data_loader(self) -> DataLoader:
        DataClass = getattr(medmnist, self.info["python_class"])
        data_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
        )

        val_dataset = DataClass(
            split="val",
            transform=data_transform,
            download=True,
            as_rgb=True,
            root=DATASET_ROOT,
        )
        val_loader = determined.pytorch.DataLoader(
            dataset=val_dataset,
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=False,
        )

        return val_loader

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int) -> Dict[str, Any]:
        inputs, targets = batch
        outputs = self.model(inputs)

        targets = torch.squeeze(targets, 1).long()
        loss = self.criterion(outputs, targets)

        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer)

        return {"loss": loss, "lr": self.lr_sch.get_last_lr()}

    def evaluate_batch(self, batch: TorchData) -> Dict[str, Any]:
        inputs, targets = batch
        outputs = self.model(inputs)
        targets = torch.squeeze(targets, 1).long()
        loss = self.criterion(outputs, targets)
        acc = torch.mean((torch.argmax(outputs, dim=1) == targets).float())
        return {"val_loss": loss, "val_accuracy": acc}
