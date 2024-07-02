# FSDP + Core API for LLM Training

This example shows how to use Fully Sharded Data Parallel [(FSDP)](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) with Determined and Core API. (Relatively) simple transformer model adapted from [GPT-fast
](https://github.com/pytorch-labs/gpt-fast) training on fake data.

## Files
* **fsdp.py**: Training setup and loop, including checkpointing, reporting, and profiling.
* **model.py**: Model architecture.
* **config.yaml**: Experiment configuration file.

## Configuration
Settings can be changed in `config.yaml` `hyperparameters` section.

### Hyperparameters
* `batch_size`: Per-device batch size.  Global batch size will be `batch_size * slots_per_trial`.
* `lr`: Learning rate.
* `d_model`, `max_seq_len`, `n_heads`, `n_layers`, `vocab_size`: Model architecture parameters.  Check code for more details.
* `report_rate`: Number of training steps to take between metric reports.
* `checkpoint_rate`: Number of training steps to take between checkpoint saves.
* `amp_dtype`: Whether to use torch automatic mixed-precision, and which dtype to use.  Options are `'auto'`, `'bfloat16'`, `'float16'`, and `null`.
* `validation_batches`: Number of batches to use when calculating validation metrics.
* `core_api_profiler`: Set to true to enable Core API profiler.  Results visible in Web UI.
* `torch_profiler`: Set to true to enable `torch` profiler.  Results visible in Tensorboard, which can be launched through the Web UI.

### Other configuration
Users might want to change `resources.slots_per_trial`, `workspace`, `project`, and `searcher.max_length` in `config.yaml`.

## Data
This example uses a synthetically generated random dataset for simplicity.

## To Run
If you have not yet installed Determined, installation instructions can be found at https://docs.determined.ai/latest/index.html

Change any desired configuration variables as outlined in the **Configuration** section, then run the following command: `det -m <master-host:port> experiment create
config.yaml .`.


## Results
Training loss should decrease from ~10.5 to ~8.5 with default settings run for 100 steps, while validation loss remains constant.  This is due to validation data being a separate random dataset.