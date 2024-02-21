# Finetuning Mistral-7B using LoRA and DeepSpeed

In this demo, we finetune the [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) using [LoRA](https://arxiv.org/abs/2106.09685) and [DeepSpeed](https://github.com/microsoft/DeepSpeed). We ran LoRA on two 80 GB A100 GPUs, and DeepSpeed on two, four, and eight 80 GB A100 GPUs.

To get started, first install Determined on your local machine:
```bash
pip install determined
```

Finetune with LoRA:
```bash
det e create distributed.yaml . 
```

Finetune with DeepSpeed:
```bash
det e create deepspeed.yaml . 
```

## Configuration

Change configuration options in `distributed.yaml`. Some important options are:
- `slots_per_trial`: the number of GPUs to use.
- `dataset_subset`: the difficulty subset to train on.
- `per_device_train_batch_size`: the batch size per GPU.


DeepSpeed configuration options are in the `ds_configs` folder.

## Testing

Test your model's generation capabilities:

```bash
python test_model.py --exp_id <exp_id> --dataset_subset <dataset_subset>
```

Where 
- `<exp_id>` is the id of your finetuning experiment in the Determined UI.
- `<dataset_subset>` is one of "easy", "medium", or "hard".

To test the pretrained model (not finetuned), leave out `--exp_id`. For example:

```bash
python test_model.py --dataset_subset easy
```

## Validating the tokenizer

Plot the distribution of dataset sample lengths, and see how many samples will be truncated by the tokenizer:

```bash
python validate_tokenizer.py
```


## Contributors

- [Kevin Musgrave](https://github.com/KevinMusgrave)
- [Agnieszka Ciborowska](https://github.com/aciborowska)