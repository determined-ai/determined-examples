# Finetuning Mistral-7B using LoRA and DeepSpeed

In this demo, we finetune [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) using [LoRA](https://arxiv.org/abs/2106.09685) and [DeepSpeed](https://github.com/microsoft/DeepSpeed). We ran LoRA on two 80 GB A100 GPUs, and DeepSpeed on two, four, and eight 80 GB A100 GPUs.

To get started, first install Determined on your local machine:
```bash
pip install determined
```

Then finetune with LoRA:
```bash
det e create lora.yaml . 
```

Or finetune with DeepSpeed:
```bash
det e create deepspeed.yaml . 
```

You can view the actual training code in `finetune.py`.




## Configuration

Change configuration options in `lora.yaml` or `deepspeed.yaml`. Some important options are:
- `slots_per_trial`: the number of GPUs to use.
- `dataset_subset`: the difficulty subset to train on.
- `per_device_train_batch_size`: the batch size per GPU.

The results in [our blog post](https://www.determined.ai/blog/llm-finetuning-2) were obtained using `per_device_train_batch_size: 1` and `per_device_eval_batch_size: 4`


DeepSpeed configuration files are in the `ds_configs` folder.

## Testing

Test your model's generation capabilities:

```bash
python inference.py --exp_id <exp_id> --dataset_subset <dataset_subset>
```

Where 
- `<exp_id>` is the id of your finetuning experiment in the Determined UI.
- `<dataset_subset>` is one of "easy", "medium", or "hard".

If you're testing a LoRA model, then add `--lora` to the above command.

To use CPU instead of GPU, add `--device cpu`.

To test the pretrained model (not finetuned), leave out `--exp_id`. For example:

```bash
python inference.py --dataset_subset easy
```

## Validating the tokenizer

Plot the distribution of dataset sample lengths, and see how many samples will be truncated by the tokenizer:

```bash
python validate_tokenizer.py
```


## Contributors

- [Kevin Musgrave](https://github.com/KevinMusgrave)
- [Agnieszka Ciborowska](https://github.com/aciborowska)
