# LLM Finetuning using HuggingFace + Determined

In this demo, we finetune the [TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v0.4) on a [text-to-SQL dataset](https://huggingface.co/datasets/Clinton/Text-to-sql-v1). We ran this on two 80 GB A100 GPUs.

To get started, first install Determined on your local machine:
```bash
pip install determined
```

Then finetune:
```bash
det e create distributed.yaml . 
```

Change configuration options in `distributed.yaml`. Some important options are:
- `slots_per_trial`: the number of GPUs to use.
- `dataset_subset`: the difficulty subset to train on.
- `per_device_train_batch_size`: the batch size per GPU.


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

## Contributors

- [Kevin Musgrave](https://github.com/KevinMusgrave)
- [Agnieszka Ciborowska](https://github.com/aciborowska)