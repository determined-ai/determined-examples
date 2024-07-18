import itertools
import json
import os
import pathlib
import time
import uuid
from typing import Optional, Callable, Dict, List, Iterable

from determined import experimental
from determined.common.api import bindings
from determined.common import context, util


def _create_task(**config) -> str:
    client = experimental.Determined()
    sess = client._session

    config_text = util.yaml_safe_dump(config)

    # TODO(ilia): try `inheritContext` instead.
    context_directory = context.read_v1_context(pathlib.Path(os.getcwd()))

    parent_id = os.environ["DET_TASK_ID"]

    req = bindings.v1CreateGenericTaskRequest(
        config=config_text,
        contextDirectory=context_directory,
        parentId=parent_id,
        # TODO(ilia): Make project id optional, inherit from the parent for child tasks.
        projectId=1,
    )

    task_resp = bindings.post_CreateGenericTask(sess, body=req)
    print(f"spawned task {task_resp.taskId}")
    return task_resp.taskId


def _wait(task_id: str, timeout: Optional[float] = 60) -> None:
    print(f"start waiting for {task_id}")
    client = experimental.Determined()
    sess = client._session

    start = time.time()

    for i in itertools.count(0):
        resp = bindings.get_GetTask(sess, taskId=task_id)
        state = resp.task.taskState
        print(state)
        TERMINAL_STATES = [
            bindings.v1GenericTaskState.COMPLETED,
            bindings.v1GenericTaskState.CANCELED,
            bindings.v1GenericTaskState.ERROR,
        ]
        if state in TERMINAL_STATES:
            return state

        if timeout > 0 and (time.time() - start > timeout):
            bindings.post_KillGenericTask(sess, taskId=task_id, body=bindings.v1KillGenericTaskRequest(taskId=task_id))
            print(f"Killed task {task_id}")
            raise RuntimeError(f"timed out waiting for task {task_id} after {i} ticks")

        time.sleep(1.)


def _dict_product(d: Dict[str, List]) -> Iterable:
    return (dict(zip(d, x)) for x in itertools.product(*d.values()))


def grid_search(train_launcher: Callable, hparam_config: Dict[str, List]):
    # Make an `external_experiment_id` for these to be displayed as an HP search.
    experiment_id = f"generic_tasks_hpsearch-{uuid.uuid4().hex[:8]}"

    # Launch a bunch of "trials".
    workers = []
    for i, hparams in enumerate(_dict_product(hparam_config)):
        trial_id = f"{i}"
        task_id = train_launcher(hparams=hparams, experiment_id=experiment_id, trial_id=trial_id)
        workers.append({
            "trial_id": trial_id,
            "task_id": task_id,
            "hparams": hparams,
        })

    sucs, errs = [], []

    # Wait for all "trials" to complete.
    for worker in workers:
        task_id, trial_id = worker["task_id"], worker["trial_id"]

        try:
            state = _wait(task_id, timeout=30)
            worker["state"] = state
        except RuntimeError:
            print(f"timed out waiting for task {task_id}")
            state = bindings.v1GenericTaskState.ERROR

        worker["state"] = state
        if state == bindings.v1GenericTaskState.COMPLETED:
            sucs.append(worker)
        else:
            errs.append(worker)

    print(f"Successful trials: {len(sucs)}, errored trials: {len(errs)}")
    if len(sucs) == 0:
        raise RuntimeError("No successful trials")

    # Inspect the trial metrics.
    client = experimental.Determined()
    sess = client._session
    max_loss = 0
    best_worker = None

    for worker in sucs:
        task_id, trial_id = worker["task_id"], worker["trial_id"]
        resp = bindings.get_GetTrialByExternalId(
            session=sess, externalExperimentId=experiment_id, externalTrialId=trial_id)
        trial = client.get_trial(resp.trial.id)
        metrics = next(trial.iter_metrics("validation"))
        loss = metrics.metrics["loss"]
        worker["loss"] = loss
        if loss > max_loss:
            max_loss = loss
            best_worker = worker

    print(f"best_worker: {best_worker}")


def train_launcher(hparams: Dict, experiment_id: str, trial_id: str) -> str:
    config = {
        "entrypoint": ["python", "train.py"],
        "resources": {
            "slots": 0,
        },
        "environment": {
            "environment_variables": [
                f"DET_HPARAMS={json.dumps(hparams)}",
                f"DET_EXTERNAL_EXPERIMENT_ID={experiment_id}",
                f"DET_EXTERNAL_TRIAL_ID={trial_id}",
            ],
        },
    }
    return _create_task(**config)


def main():
    hparams_config = {
        "x": [1, 2],
        "y": [3, 4],
    }

    grid_search(train_launcher=train_launcher, hparam_config=hparams_config)

    print("Done")


if __name__ == "__main__":
    main()
