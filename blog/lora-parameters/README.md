# Finding the best LoRA parameters

We finetune [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) using [LoRA](https://arxiv.org/abs/2106.09685) and [DeepSpeed](https://github.com/microsoft/DeepSpeed). We ran LoRA on two 40 GB A100 GPUs utilizing DeepSpeed.  

See our [blog post](https://www.determined.ai/blog/lora-parameters) for our experiment results.

To get started, first install Determined on your local machine:
```bash
pip install determined
```

Then finetune with LoRA:
```bash
det e create lora.yaml . 
```

You can view the actual training code in `finetune.py`.


## Configuration

Change configuration options in `lora.yaml`. Some important options are:
- `slots_per_trial`: the number of GPUs to use.
- `dataset_subset`: the difficulty subset to train on.
- `per_device_train_batch_size`: the batch size per GPU.


DeepSpeed configuration files are in the `ds_configs` folder.


## Contributors

- By [Sze Wai Yuen](https://github.com/szewaiyuen6)
- Built on `llm-finetuning` code by [Agnieszka Ciborowska](https://github.com/aciborowska) and [Kevin Musgrave](https://github.com/KevinMusgrave).