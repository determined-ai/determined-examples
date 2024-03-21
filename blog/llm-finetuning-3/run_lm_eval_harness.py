import fnmatch
import json
import logging
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import determined as det
import numpy as np
from lm_eval import base, evaluator, models, tasks
from transformers.trainer_utils import get_last_checkpoint


def get_lm(
    core_context: det.core.Context,
    hparams: Dict[str, Any],
    model: str = "hf-causal-experimental",
):
    # Extract model args.
    model_args = [
        f"trust_remote_code={hparams['model_args']['trust_remote_code']}",
        f"use_accelerate={hparams['model_args']['use_accelerate']}",
    ]

    if "hf_model_name" in hparams["model_args"]:
        model_args.append(f"pretrained={hparams['model_args']['hf_model_name']}")

    if "token" in hparams["model_args"]:
        import huggingface_hub

        huggingface_hub.login(token=hparams["model_args"]["token"])

    model_args_str = ",".join(model_args)

    # Load from det checkpoint or build from HF.
    if "model_ckpt" in hparams["model_args"]:
        model_ckpt = hparams["model_args"]["model_ckpt"]
        logging.info(f"Loading model from checkpoint {model_ckpt}")
        with core_context.checkpoint.restore_path(model_ckpt) as path:
            lm = models.get_model(model).create_from_arg_string(
                model_args_str,
                {
                    "batch_size": hparams["batch_size"],
                    "device": hparams["device"],
                    "pretrained": get_last_checkpoint(path),
                },
            )
    elif model_args_str is not None:
        lm = models.get_model(model).create_from_arg_string(
            model_args_str,
            {
                "batch_size": hparams["batch_size"],
                "max_batch_size": hparams["max_batch_size"],
                "device": hparams["device"],
            },
        )
        lm = base.CachingLM(
            lm,
            "lm_cache/"
            + model
            + "_"
            + model_args_str.replace("=", "-").replace(",", "_").replace("/", "-")
            + ".db",
        )
    return lm


def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)


def get_task_dict(hparams: Dict[str, Any]) -> Dict[str, base.Task]:
    """
    Helper function to create dict of tasks, as expected by the harness.

    hparams["task"] contains a single string defining one or more (if the str is a
    glob pattern) tasks.
    """
    if isinstance(hparams["task"], dict):
        task_list: List[str] = pattern_match([hparams["task"]["name"]], tasks.ALL_TASKS)
        task_dict = tasks.get_task_dict(task_list)
        return task_dict
    else:
        raise ValueError(
            f"Received task of type {type(hparams['task'])}, expected str or dict"
        )


def run(core_context: det.core.Context, hparams: Dict[str, Any]) -> None:
    random.seed(1234)
    np.random.seed(1234)
    task_dict = get_task_dict(hparams)
    lm = get_lm(core_context, hparams)

    results = evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        num_fewshot=hparams["task"]["num_fewshot"],
    )

    all_metrics = _aggregate_metrics(results, hparams)
    _report_metrics(core_context, results, all_metrics)


def _aggregate_metrics(results, hparams: Dict[str, Any]) -> Dict[str, Any]:
    # Report all tasks in separate metrics groups, with a meta group for the average metrics across
    # tasks.
    all_metrics: Dict[str, Dict[str, Union[List[float], float]]] = defaultdict(dict)  # type: ignore
    # If more than one result was reported, then a glob pattern was used and we report the average
    # across all matching tasks.
    glob_pattern: Optional[str] = hparams["task"] if len(results["results"]) > 1 else None  # type: ignore
    avg_glob_key: Optional[str] = glob_pattern + "_average" if glob_pattern is not None else None  # type: ignore

    for task_name, metrics in results["results"].items():
        for metric_name, value in metrics.items():
            all_metrics[task_name][metric_name] = value

            # Compute averages across tasks if more than one task was performed.
            if glob_pattern:
                if metric_name not in all_metrics[avg_glob_key]:
                    all_metrics[avg_glob_key][metric_name] = []
                assert isinstance(all_metrics[avg_glob_key][metric_name], list)
                all_metrics[avg_glob_key][metric_name].append(value)

    if avg_glob_key is not None:
        for k, v in all_metrics[avg_glob_key].items():
            assert isinstance(v, list)
            all_metrics[avg_glob_key][k] = sum(v) / len(v)

    return all_metrics


def _report_metrics(core_context: det.core.Context, results, all_metrics: Dict) -> None:
    dumped = json.dumps(results, indent=2)
    print(dumped)

    for group, metrics in all_metrics.items():
        core_context.train.report_metrics(
            group=group, steps_completed=info.trial.trial_id, metrics=metrics
        )
    with core_context.checkpoint.store_path({"steps_completed": 1}) as (
        path,
        uuid,
    ):
        with open(path.joinpath("results.txt"), "w") as f:
            f.write(dumped)

    print(evaluator.make_table(results))


if __name__ == "__main__":
    info = det.get_cluster_info()
    assert info is not None
    hparams = info.trial.hparams

    with det.core.init() as core_context:
        run(core_context, hparams)
