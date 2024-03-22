import os
import json

from determined.experimental import core_v2

def main():
    hparams = json.loads(os.environ["DET_HPARAMS"])
    x, y = hparams["x"], hparams["y"]
    experiment_id = os.environ["DET_EXTERNAL_EXPERIMENT_ID"]
    trial_id = os.environ["DET_EXTERNAL_TRIAL_ID"]

    core_v2.init(
        defaults=core_v2.DefaultConfig(
            name="generic_tasks_hpsearch",
            hparams=hparams,
            searcher={
                "name": "custom",
                "metric": "loss",
                "smaller_is_better": False,
            },
        ),
        unmanaged=core_v2.UnmanagedConfig(
            external_experiment_id=experiment_id,
            external_trial_id=trial_id,
        ),
    )

    loss = x * y
    core_v2.train.report_validation_metrics(steps_completed=0, metrics={"loss": loss})


if __name__ == "__main__":
    main()
