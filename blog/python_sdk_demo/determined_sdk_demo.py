"""This script shows example usage of the Determined Python SDK to run experiments.

It:
1. Archives any existing experiments with the same names as the datasets we'll train on.
2. Creates models for each dataset and registers them in the Determined model registry.
3. Trains a model for each dataset by creating an experiment.
4. Registers the best checkpoint for each experiment in the Determined model registry.

For an in-depth discussion of this script, see the blog post:
    https://www.determined.ai/blog/python-sdk

For more information on the Determined Python SDK, see:
    https://docs.determined.ai/latest/reference/python-sdk.html
"""
from typing import Dict, List, Optional

from determined.common.api import errors
from determined.experimental import client
import medmnist
import yaml


WORKSPACE = "SDK Demo"  # The workspace that contains the projects
PROJECT = "MedMNIST"  # The project that contains experiments
MODEL_DIR = "mednist_model"  # Where the config and model_def files live
# We'll train models on the these 3 MedMNIST datasets
DATASETS = ["dermamnist", "bloodmnist", "retinamnist"]
DEMO_VERSION = "demoV1"


def setup_projects(workspace_name: str, project_names: List[str]) -> None:
    """Create projects in a workspace if they don't already exist.

    Projects belong to one workspace. One workspace can have many projects.

    Args:
        workspace_name: Name of the workspace to create.
        project_names: List of project names to create in the passed workspace.
    """
    try:
        workspace = client.get_workspace(workspace_name)
    except errors.NotFoundException:
        print(f"Creating workspace '{workspace_name}'")
        workspace = client.create_workspace(workspace_name)

    workspace_project_names = [project.name for project in workspace.list_projects()]
    for name in project_names:
        if name not in workspace_project_names:
            print(f"Creating project '{name}'")
            workspace.create_project(name)


def archive_experiments(
    experiment_names: List[str], workspace_name: str, project_name: str
) -> None:
    """Archive any existing, ended experiments with the same names as the datasets we train on.

    Experiment names are not unique, so this function may result in archiving several experiments
    for each passed name. Experiments with matching names that are not yet done are left alone.

    Projects are used to organize experiments. Workspaces organize projects.

    Args:
        experiment_names: List of experiment names.
        workspace_name: Name of the workspace that contains the experiments.
        project_name: Name of the project that contains the experiments.
    """
    project_id = client.get_workspace(workspace_name).get_project(project_name).id

    for name in experiment_names:
        exps = client.list_experiments(name=name, project_id=project_id)
        for exp in exps:
            if not exp.archived:
                if exp.state.value in (
                    client.ExperimentState.COMPLETED.value,
                    client.ExperimentState.CANCELED.value,
                    client.ExperimentState.DELETED.value,
                    client.ExperimentState.ERROR.value,
                ):
                    print(f"Archiving experiment {exp.id} (dataset={exp.name})")
                    exp.archive()
                else:
                    print(
                        f"Not archiving experiment {exp.id} (dataset={exp.name}) because it is"
                        f" still in state {exp.state}"
                    )


def create_model(name: str, workspace: str) -> None:
    """Create models for each dataset and register them in the Determined model registry.

    If a model of the passed name already exists, this function moves it to the passed workspace
    if necessary.

    Args:
        model_name: Name of the model to create.
        workspace_name: Name of the workspace that contains the models.
    """
    workspace_id = client.get_workspace(workspace).id
    try:
        model = client.get_model(name)
        print(f"Using existing model '{name}' from the registry")
    except errors.NotFoundException:
        print(f"Creating new model '{name}' in the registry")
        model = client.create_model(name=name)

    if model.workspace_id != workspace_id:
        model.move_to_workspace(workspace_name=workspace)


def run_experiment(
    dataset: str, workspace: str, project: str, labels: Optional[str]
) -> client.Experiment:
    """Run an experiment for a dataset.

    Args:
        dataset: MedMNIST dataset name for this experiment.
        workspace: Name of the workspace to run the experiments in.
        project: Name of the project to run the experiments in.
        labels: Optional list of labels to tag the experiments with.

    Returns:
        An Experiment object representing the experiments spawned on the Determined platform.
    """
    with open(f"{MODEL_DIR}/config.yaml", "r") as file:
        exp_conf: Dict[str, str] = yaml.safe_load(file)

    exps = []

    # Set configuration particular to this dataset and example script
    exp_conf["name"] = dataset
    exp_conf["workspace"] = workspace
    exp_conf["project"] = project
    exp_conf["records_per_epoch"] = medmnist.INFO[dataset]["n_samples"]["train"]
    exp_conf["hyperparameters"]["data_flag"] = dataset

    print(f"Starting experiment for dataset {dataset}")
    exp = client.create_experiment(config=exp_conf, model_dir=MODEL_DIR)
    print(f"Experiment {dataset} started with id {exp.id}")

    for label in labels:
        exp.add_label(label)

    return exp


def finish_experiment(exp: client.Experiment) -> client.Checkpoint:
    """Wait for an experiment to complete and clean it up.

    This function:
    1. Waits for an experiment to reach a terminal state
    2. If it completed successfully, it finds and returns the best checkpoint

    Args:
        exp: An Experiment object

    Returns:
        The best Checkpoint per the experiment's "searcher metric".  In this example, the experiment
        config specifies "val_accuracy" as the searcher metric.

    Raises:
        RuntimeError: If the experiment did not complete successfully.
    """
    exit_status = exp.wait()
    print(f"Experiment {exp.id} completed with status {exit_status}")
    if exit_status == client.ExperimentState.COMPLETED:
        checkpoints = exp.list_checkpoints(
            max_results=1,
            sort_by=client.CheckpointSortBy.SEARCHER_METRIC,
            order_by=client.OrderBy.DESCENDING,
        )

        return checkpoints[0]
    else:
        raise RuntimeError(
            f"Experiment {exp.name} (id={exp.id}) did not complete successfully."
            f" It is currently in state {exp.state}"
        )


def main():
    client.login()  # Host address & user credentials can be optionally passed here

    setup_projects(
        workspace_name=WORKSPACE,
        project_names=[PROJECT],
    )
    archive_experiments(
        experiment_names=DATASETS,
        workspace_name=WORKSPACE,
        project_name=PROJECT,
    )
    exps = []
    for dataset in DATASETS:
        create_model(name=dataset, workspace=WORKSPACE)
        exps.append(
            run_experiment(dataset, workspace=WORKSPACE, project=PROJECT, labels=[DEMO_VERSION])
        )  # Run the experiments in parallel

    print("Waiting for experiments to complete...")
    for exp in exps:
        best_checkpoint = finish_experiment(exp)
        # models and experiments are both named after their medmnist dataset
        model = client.get_model(exp.name)
        model_version = model.register_version(best_checkpoint.uuid)
        model_version.set_notes(f"Creating using Determined SDK demo version {DEMO_VERSION}")


if __name__ == "__main__":
    main()
