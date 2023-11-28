# CycleGAN

Determined implementation of [CycleGAN](https://github.com/eriklindernoren/PyTorch-GAN#cyclegan)

## Important Files

- **determined_model_def.py** is the user model definition file for Determined-managed training.
  This file is ported from **cyclegan.py** to implement the
  [Determined Pytorch API](https://docs.determined.ai/latest/reference/api/pytorch.html#pytorch-trial).
- **const.yaml** is the user configuration for Determined-managed training.
- **startup-hook.sh** is the startup bash script that is used for downloading and
  extracting the training data.
- **cyclegan.py** is the original training script in the original repository.

## To Run

**Prerequisites**: A Determined cluster must be installed in order to run this example.
Please follow the directions [here](https://docs.determined.ai/latest/how-to/install-main.html)
in order to install.

To run the example, simply submit the experiment to the cluster by running the
following command from this directory:

`det -m <master host:port> experiment create 1-gpu.yaml . `

## Modify the configuration

### Distributed Training

1. Change resources .

```yaml
resources:
  slots_per_trial: 64
```

2. Change global batch sizes

```yaml
hyperparameters:
  global_batch_size: 64
```

## Expected Performance

| Who Manages Training | GPU Type | GPU Number Per Node | Node Number | Global Batch Size | Aggregation Frequency | Throughput                   |
| -------------------- | -------- | ------------------- | ----------- | ----------------- | --------------------- | ---------------------------- |
| User                 | V100     | 1                   | 1           | 1                 | 1                     | 4.948 records / sec          |
| Determined           | V100     | 1                   | 1           | 1                 | 1                     | 4.878 records / sec          |
| Determined           | V100     | 8                   | 8           | 64                | 1                     | 71.44-164.45 records / sec\* |
| Determined           | V100     | 8                   | 8           | 64                | 2                     | 157.18 records / sec         |
| Determined           | V100     | 8                   | 8           | 64                | 4                     | 157.18 records / sec         |
| Determined           | V100     | 8                   | 8           | 128               | 1                     | 232.85 records / sec         |
| Determined           | V100     | 8                   | 8           | 256               | 1                     | 299.38 records / sec         |

- The throughput is unstable due to inter-node communication when the global batch size
  is 64 and the aggregation frequency is 1. Use a larger batch size or a larger aggregation
  frequency to increase the scaling efficiency of the throughput. See
  [Effective Distributed Training](https://docs.determined.ai/latest/topic-guides/effective-distributed-training.html#effective-distributed-training)
  for details.
