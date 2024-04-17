# Finetuning Gemma-2B using DPO

In this demo, we train the [Gemma-2B](https://huggingface.co/google/gemma-2b) model to both understand 
instructions and produce outputs that align with human preferences. 
We start with supervised fine-tuning (SFT) it with an instruction dataset so that the model gets better at answering user queries. 
Next, we use Direct Preference Optimization (DPO) and a dataset of human preferences
to make the model output align better with human preferences.

To get started, first install Determined on your local machine:
```bash
pip install determined
```

To perform SFT step:
```bash
det e create sft.yaml . 
```
To perform DPO step:

```bash
det e create dpo.yaml . 
```

You can view the actual training code in `sft_finetune.py` and `dpo_finetune.py`.

## Configuration

Change configuration options in `sft.yaml` or `dpo.yaml`. Some important options are:
- `slots_per_trial`: the number of GPUs to use.
- `per_device_train_batch_size`: the batch size per GPU.
- `training_args`: training parameters such as learning rate or the number of training epochs.

For SFT step, you can also set:
- `dataset_subsets`: update ratios of datasets.
- `data_collator`: turns on/off training on completions only.
- `chat_tokens`: adds special chat tokens to the vocabulary.
- `max_seq_length`: sets maximum sequence length.

For DPO step, you have the following options:
- `dpo_beta`: sets beta value for DPO training.
- `dpo_loss`: one of the available dpo losses to use: "sigmoid", "hinge", "cdpo", "ipo", "kto_pair". 
You can learn more about it [here](https://huggingface.co/docs/trl/main/en/dpo_trainer#loss-functions).
- `max_length`, `max_prompt_length`, and `max_target_length`: sets maximum sequence length for 
combined prompt and target, prompt, target, respectively.

## Testing

Test your model's generation capabilities, you can use the inference script that generates outputs for
a number of samples from [Intel/orca_dpo_pairs](https://huggingface.co/datasets/Intel/orca_dpo_pairs) dataset:
```bash
python inference.py [--exp_id <exp_id>] [--trial_id <trial_id>] [--output_file <filename>] [--number_of_samples n]
```

Where 
- `<exp_id>` is the id of your finetuning experiment in the Determined UI. Use either `--exp_id` or `--trial_id`.
- `<trial_id>` is the id of your finetuning trial in the Determined UI. Use either `--exp_id` or `--trial_id`.
- `<filename>` is the csv file with model generations.
- `<n>` is the number of samples from the dataset you want to use for generation. 
To use CPU instead of GPU, add `--device cpu`.

To test the pretrained model (not finetuned), leave out `--exp_id` and `--trial_id`. For example:

```bash
python inference.py
```


## Contributors

- [Agnieszka Ciborowska](https://github.com/aciborowska)
- [Kevin Musgrave](https://github.com/KevinMusgrave)

