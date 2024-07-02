import json
import logging
import os
import random
from typing import Any, Dict, Generator, Optional, TypedDict

import determined as det
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from model import EmbedAndEncode, LMHead, Transformer, TransformerBlock

"""
Minimal transformer model FSDP script with Core API.
"""


def get_fake_data_iter(
    batch_size: int,
    vocab_size: int,
    max_seq_len: int,
    rank: int,
    device: torch.device,
    is_validation: bool,
    simulated_size_in_batches: int = 10,
) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
    """
    Fake dataloader. Yields a different set of data for each rank, and for train vs validation.
    This data would usually come from a tokenized dataset.
    """
    generator = torch.Generator(device=device)
    next_idx = 0
    while True:
        if next_idx == 0:
            generator.manual_seed(42 + rank + 100000 * is_validation)
        fake_sequence = torch.randint(
            vocab_size,
            (batch_size, max_seq_len + 1),
            device=device,
            generator=generator,
        )
        inputs, targets = fake_sequence[..., :-1], fake_sequence[..., 1:]
        yield inputs, targets
        next_idx = (next_idx + 1) % simulated_size_in_batches


def get_loss(
    fsdp_model: FSDP, batch: tuple[torch.Tensor, torch.Tensor], use_amp: bool
) -> torch.Tensor:
    inputs, labels = batch
    with torch.cuda.amp.autocast(enabled=use_amp):
        outputs = fsdp_model(inputs)
        outputs_flat = outputs.reshape(-1, outputs.shape[-1])
        labels_flat = labels.reshape(-1)
        loss = F.cross_entropy(outputs_flat, labels_flat)
    return loss


def get_reduced_loss_and_report(
    loss_history: list[torch.Tensor],
    steps_completed: int,
    core_context: det.core.Context,
    validation: bool,
) -> Optional[float]:
    """
    Average the most recent losses across all processes and report the result.  Returns the reduced
    loss on rank 0 and None on all other ranks.
    """

    loss_history_t = torch.stack(loss_history).mean()
    dist.reduce(loss_history_t, 0, op=dist.ReduceOp.AVG)
    if core_context.distributed.rank == 0:
        reduced_loss = loss_history_t.item()
        # TypedDict pattern to satisfy mypy.
        ReportArgs = TypedDict(
            "ReportArgs", {"steps_completed": int, "metrics": Dict[str, float]}
        )
        report_args: ReportArgs = {
            "steps_completed": steps_completed,
            "metrics": {"loss": reduced_loss},
        }
        if validation:
            core_context.train.report_validation_metrics(**report_args)
        else:
            core_context.train.report_training_metrics(**report_args)
        return reduced_loss
    return None


def save_checkpoint(
    fsdp_model: FSDP,
    optimizer: torch.optim.Optimizer,
    scaler: ShardedGradScaler,
    use_amp: bool,
    core_context: det.core.Context,
    steps_completed: int,
) -> None:
    # All ranks collectively build the checkpoint on rank 0:

    with FSDP.state_dict_type(
        fsdp_model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        model_state_dict = fsdp_model.state_dict()
        optim_state_dict = FSDP.optim_state_dict(fsdp_model, optimizer)

    if core_context.distributed.rank == 0:
        with core_context.checkpoint.store_path(
            metadata={"steps_completed": steps_completed}
        ) as (
            path,
            _,
        ):
            torch.save(model_state_dict, path.joinpath("model.bin"))
            torch.save(optim_state_dict, path.joinpath("optim.bin"))
            if use_amp:
                # Scaler state is automatically the same across ranks.
                scaler_state_dict = scaler.state_dict()
                torch.save(scaler_state_dict, path.joinpath("scaler.bin"))


def load_checkpoint(
    fsdp_model: FSDP,
    optimizer: torch.optim.Optimizer,
    scaler: ShardedGradScaler,
    use_amp: bool,
    core_context: det.core.Context,
    device: torch.device,
    uuid: str,
) -> int:
    with core_context.checkpoint.restore_path(uuid) as path:
        with FSDP.state_dict_type(
            fsdp_model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            fsdp_model.load_state_dict(
                torch.load(path.joinpath("model.bin"), map_location=device)
            )
            optim_state_dict = torch.load(
                path.joinpath("optim.bin"), map_location=device
            )
            optim_state_dict_to_load = FSDP.optim_state_dict_to_load(
                model=fsdp_model,
                optim=optimizer,
                optim_state_dict=optim_state_dict,
            )
            optimizer.load_state_dict(optim_state_dict_to_load)
        scaler_path = path.joinpath("scaler.bin")
        if use_amp and os.path.isfile(scaler_path):
            scaler.load_state_dict(torch.load(scaler_path))

        with open(path.joinpath("metadata.json"), "r") as f:
            metadata = json.load(f)

    last_step_completed = metadata["steps_completed"]
    return last_step_completed


def main(
    core_context: det.core.Context,
    hparams: dict[str, Any],
    checkpoint_uuid: Optional[str] = None,
) -> None:
    # Fix the random seed on all devices
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Get and set the device for this process
    device = torch.device(f"cuda:{core_context.distributed.local_rank}")
    torch.cuda.set_device(device)

    # Build the unsharded model directly on the device.
    model = Transformer(
        d_model=hparams["d_model"],
        n_heads=hparams["n_heads"],
        vocab_size=hparams["vocab_size"],
        n_layers=hparams["n_layers"],
        max_seq_len=hparams["max_seq_len"],
        device=device,
    )

    # Inspect the model:
    if core_context.distributed.rank == 0:
        print("Model before FSDP:")
        print(model, flush=True)

    # Wrap the embedding layer, the lm head, and each transformer block into its own FSDP unit:
    auto_wrap_policy = ModuleWrapPolicy([TransformerBlock, EmbedAndEncode, LMHead])

    # The fsdp model:
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device,
        use_orig_params=True,
    )

    # Inspect the model post-FSDP
    if core_context.distributed.rank == 0:
        print("Model after FSDP:")
        print(fsdp_model, flush=True)

    # The optimizer must be created post-FSDP
    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=hparams["lr"])

    steps_completed = 0
    report_rate = hparams["report_rate"]
    checkpoint_rate = hparams["checkpoint_rate"]
    validation_batches = hparams["validation_batches"]
    use_amp = hparams["use_amp"]
    use_torch_profiler = hparams["torch_profiler"]
    train_loss_history = []

    data_iter_arguments = {
        "batch_size": hparams["batch_size"],
        "vocab_size": hparams["vocab_size"],
        "max_seq_len": hparams["max_seq_len"],
        "rank": core_context.distributed.rank,
        "device": device,
    }
    train_data_iter = get_fake_data_iter(is_validation=False, **data_iter_arguments)
    scaler = ShardedGradScaler(enabled=use_amp)
    # If a previous checkpoint exists, load it now and correct the steps_completed:
    if checkpoint_uuid is not None:
        steps_completed = load_checkpoint(
            fsdp_model,
            optimizer,
            scaler,
            use_amp,
            core_context,
            device,
            checkpoint_uuid,
        )
    # If torch profiler enabled, write profiling results to TensorBoard accessible through WebUI.
    if use_torch_profiler:
        torch_profiler = torch.profiler.profile(
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(core_context.train.get_tensorboard_path())
            ),
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
        )
    for op in core_context.searcher.operations():
        # Train for the number of steps specified in searcher.max_length in config.yaml
        while steps_completed < op.length:
            batch = next(train_data_iter)
            loss = get_loss(fsdp_model, batch, use_amp)
            train_loss_history.append(loss.detach().clone())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if use_torch_profiler:
                torch_profiler.step()

            steps_completed += 1
            this_is_the_last_step = steps_completed == op.length

            if steps_completed % report_rate == 0 or this_is_the_last_step:
                # Report the average training loss.
                get_reduced_loss_and_report(
                    train_loss_history, steps_completed, core_context, validation=False
                )
                train_loss_history.clear()
                # Compute and report an average validation loss.
                validation_data_iter = get_fake_data_iter(
                    is_validation=True, **data_iter_arguments
                )
                validation_loss_history = []
                with torch.inference_mode():
                    for i in range(validation_batches):
                        batch = next(validation_data_iter)
                        loss = get_loss(fsdp_model, batch, use_amp)
                        validation_loss_history.append(loss)
                last_validation_loss = get_reduced_loss_and_report(
                    validation_loss_history,
                    steps_completed,
                    core_context,
                    validation=True,
                )

            if steps_completed % checkpoint_rate == 0 or this_is_the_last_step:
                save_checkpoint(
                    fsdp_model,
                    optimizer,
                    scaler,
                    use_amp,
                    core_context,
                    steps_completed,
                )
                # Since should_preempt is blocking, we only check at checkpoint_rate to
                # maintain performance.
                if core_context.preempt.should_preempt():
                    return

        # Tell the master we're done
        if core_context.distributed.rank == 0:
            op.report_completed(last_validation_loss)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format=det.LOG_FORMAT)
    info = det.get_cluster_info()
    assert info, "This script must run on a determined cluster."
    assert torch.cuda.is_available(), "This script assumes cuda."

    checkpoint_uuid = info.latest_checkpoint
    hparams = info.trial.hparams
    core_api_profiler = hparams["core_api_profiler"]
    try:
        dist.init_process_group("nccl")
        distributed = det.core.DistributedContext.from_torch_distributed()
        with det.core.init(distributed=distributed) as core_context:
            if core_api_profiler:
                core_context.profiler.on()
            main(
                core_context=core_context,
                hparams=hparams,
                checkpoint_uuid=checkpoint_uuid,
            )
            if core_api_profiler:
                core_context.profiler.off()
    finally:
        dist.destroy_process_group()
