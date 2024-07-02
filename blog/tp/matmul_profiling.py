import gc
import logging

import determined as det
import torch

import utils

"""
Script for profiling square matmuls on a single GPU.
"""


def profile_and_report(
    core_context: det.core.Context,
    d_model: int,
    num_repeats: int,
    num_warmups: int,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    A = torch.randn(d_model, d_model, device="cuda", dtype=dtype)
    B = torch.randn(d_model, d_model, device="cuda", dtype=dtype)

    # Use CUDA events for accurate timing.
    timer = utils.CUDAEventTimer()
    torch.cuda.synchronize()

    # Warmups
    for _ in range(num_warmups):
        A @ B

    # Timed region.
    for _ in range(num_repeats):
        with timer:
            A @ B

    # Mean and std TFLOP computations
    flops = 2 * d_model**3
    time_s_t = torch.tensor(timer.time_s_list)
    tflop_s_gpu_t = flops / time_s_t / 1e12
    metrics = {
        "d_model": d_model,
        "time_s": timer.time_s_mean,
        "time_s_std": timer.time_s_std,
        "tflop_s_gpu": tflop_s_gpu_t.mean().item(),
        "tflop_s_gpu_std": tflop_s_gpu_t.std().item(),
    }

    #  Use d_model as the x-axis for plotting purposes.
    core_context.train.report_metrics(group="matmul", steps_completed=d_model, metrics=metrics)

    # Memory management
    del A
    del B
    gc.collect()
    torch.cuda.empty_cache()


def main(
    core_context: det.core.Context,
    d_model_min: int,
    d_model_max: int,
    d_model_step: int,
    num_repeats: int,
    num_warmups: int,
) -> None:
    for d_model in range(d_model_min, d_model_max + 1, d_model_step):
        profile_and_report(
            core_context=core_context,
            d_model=d_model,
            num_repeats=num_repeats,
            num_warmups=num_warmups,
        )


if __name__ == "__main__":
    info = det.get_cluster_info()
    assert info, "This script must run on a determined cluster."
    hparams = info.trial.hparams

    with det.core.init() as core_context:
        logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)

        main(
            core_context=core_context,
            d_model_min=hparams["d_model_min"],
            d_model_max=hparams["d_model_max"],
            d_model_step=hparams["d_model_step"],
            num_repeats=hparams["num_repeats"],
            num_warmups=hparams["num_warmups"],
        )
