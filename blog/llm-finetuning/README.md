# LLM Finetuning using HuggingFace + Determined

First install Determined on your local machine:
```bash
pip install determined
```

Then finetune:
```bash
det e create distributed.yaml . 
```

Change configuration options in `distributed.yaml`.

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