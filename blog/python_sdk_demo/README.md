# det-python-sdk-demo

## Overview

This script shows example usage of the Determined Python SDK to run and administer experiments.

It:
1. Archives any existing experiments with the same names as the datasets we'll train on.
2. Creates models for each dataset and registers them in the Determined model registry.
3. Trains a model for each dataset by creating an experiment.
4. Registers the best checkpoint for each experiment in the Determined model registry.

For an in-depth discussion of this script, see the blog post:
    https://www.determined.ai/blog/python-sdk

For more information on the Determined Python SDK, see:
    https://docs.determined.ai/latest/reference/python-sdk.html

## Installation / Execution

To run this demo:

1. Install dependencies. In addition to the determined CLI, we this demo uses MedMNIST datasets.
```
pip install -r requirements.txt
```

2. Set DET_MASTER environment variable. For example, if you're running this locally:
```
export DET_MASTER=localhost:8080
```

For more information about configuring the CLI, see [this doc](https://docs.determined.ai/latest/setup-cluster/setup-clients.html#setting-up-clients).

3. Now the demo is ready to be executed. To run experiments:
```
python determined_sdk_demo.py
```

## Contributors

- [Wesley Turner](https://github.com/wes-turner)
- [Kevin Musgrave](https://github.com/KevinMusgrave)

The code in the `medmnist_model` directory is based on the [`determined_medmnist_e2e`](https://github.com/ighodgao/determined_medmnist_e2e) repo by [Isha Ghodgaonkar](ighodgao).