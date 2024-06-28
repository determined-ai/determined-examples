import gc
import logging
import os

import determined as det
import torch
import torch.distributed as dist

import layers
import utils

"""
Script for profiling the forward pass of TP MLP layers. Measures the iteration time and computes the
TFLOPs/sec/GPU for all availbable MLP configurations sharded across power-of-two GPUs (including
including the single GPU, non-TP case).

Only intended for single-node use.
"""


def profile_and_report(
    core_context: det.core.Context,
    batch_size: int,
    seq_len: int,
    d_model: int,
    num_repeats: int,
    num_warmups: int,
    device: torch.device,
    rank: int,
    tp_degree: int,
    pg_dict: dict[int, dist.ProcessGroup],
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    # This rank doesn't participate if it's not in the TP group.
    if rank >= tp_degree:
        return

    inputs = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
    if tp_degree == 1:
        mlp = layers.MLP(d_model=d_model, dtype=dtype, device=device)
    else:
        mlp = layers.MLPTP(d_model=d_model, dtype=dtype, device=device, group=pg_dict[tp_degree])

    # Use CUDA events for accurate timing.
    timer = utils.CUDAEventTimer()

    # Warmups
    for _ in range(num_warmups):
        mlp(inputs)

    # Timed region.
    for _ in range(num_repeats):
        with timer:
            mlp(inputs)

    # Mean and std TFLOP computations
    mlp_flops = 16 * batch_size * seq_len * d_model**2
    time_s_t = torch.tensor(timer.time_s_list)
    tflop_s_gpu_t = mlp_flops / time_s_t / 1e12 / tp_degree
    metrics = {
        "d_model": d_model,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "tp_degree": tp_degree,
        "time_s": timer.time_s_mean,
        "time_s_std": timer.time_s_std,
        "tflop_s_gpu": tflop_s_gpu_t.mean().item(),
        "tflop_s_gpu_std": tflop_s_gpu_t.std().item(),
    }

    # report metrics on rank zero. Use d_model as the x-axis for plotting purposes.
    if not rank:
        core_context.train.report_metrics(
            group=f"tp_degree_{tp_degree}", steps_completed=d_model, metrics=metrics
        )

    # Memory management
    del mlp
    del inputs
    gc.collect()
    torch.cuda.empty_cache()


def main(
    core_context: det.core.Context,
    batch_size: int,
    seq_len: int,
    d_model_min: int,
    d_model_max: int,
    d_model_step: int,
    num_repeats: int,
    num_warmups: int,
) -> None:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Profile every possible power-of-2 TP group size
    def is_power_of_two(n: int) -> bool:
        return n & (n - 1) == 0

    tp_degrees = [n for n in range(1, world_size + 1) if is_power_of_two(n)]
    # Create the non-trivial process groups
    pg_dict = {
        tp_degree: dist.new_group(list(range(tp_degree)), backend="nccl")
        for tp_degree in tp_degrees
        if tp_degree > 1
    }

    for tp_degree in tp_degrees:
        for d_model in range(d_model_min, d_model_max + 1, d_model_step):
            dist.barrier()
            torch.cuda.synchronize()
            profile_and_report(
                core_context=core_context,
                batch_size=batch_size,
                seq_len=seq_len,
                d_model=d_model,
                num_repeats=num_repeats,
                num_warmups=num_warmups,
                rank=rank,
                tp_degree=tp_degree,
                pg_dict=pg_dict,
                device=device,
            )


if __name__ == "__main__":
    info = det.get_cluster_info()
    assert info, "This script must run on a determined cluster."
    hparams = info.trial.hparams

    # Set up determined's distributed code, if needed
    try:
        distributed = det.core.DistributedContext.from_torch_distributed()
        dist.init_process_group("nccl")
    except KeyError:
        distributed = None

    try:
        with det.core.init(distributed=distributed) as core_context:
            logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)

            main(
                core_context=core_context,
                batch_size=hparams["batch_size"],
                seq_len=hparams["seq_len"],
                d_model_min=hparams["d_model_min"],
                d_model_max=hparams["d_model_max"],
                d_model_step=hparams["d_model_step"],
                num_repeats=hparams["num_repeats"],
                num_warmups=hparams["num_warmups"],
            )
    finally:
        if distributed is not None:
            dist.destroy_process_group()
