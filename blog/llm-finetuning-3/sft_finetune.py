import logging
import sys
from typing import List

import datasets
import determined as det
import torch
import transformers
from datasets import concatenate_datasets, load_dataset
from determined.transformers import DetCallback
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, PreTrainedModel,
                          PreTrainedTokenizer, TrainingArguments)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer, setup_chat_format

from chat_format import CHAT_ML_TEMPLATE, get_response_template_ids

logger = logging.getLogger(__name__)


def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 2048

    if tokenizer.chat_template is None and tokenizer.default_chat_template is None:
        tokenizer.chat_template = CHAT_ML_TEMPLATE

    tokenizer.add_eos_token = True
    return tokenizer


def get_model_and_tokenizer(model_name, use_lora, inference=False, device_map="auto"):
    if inference:
        if use_lora:
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map=device_map
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )

        if use_lora:
            peft_config = LoraConfig(
                task_type="CAUSAL_LM",
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
            )

            model = get_peft_model(model, peft_config)

    tokenizer = get_tokenizer(model_name)
    return model, tokenizer


def load_sft_dataset(hparams):
    dataset_name = hparams["dataset"]
    dataset_subsets = hparams["dataset_subsets"]
    dataset_list = []
    for subset_info in dataset_subsets:
        dataset_subset = load_dataset(dataset_name, subset_info["subset"])["train"]
        if "ratio" in subset_info:
            number_of_samples = int(len(dataset_subset) * subset_info["ratio"])
        elif "number_of_samples" in subset_info:
            number_of_samples = subset_info["number_of_samples"]
        else:
            raise RuntimeError(f"Unknown subset definition {subset_info}")
        dataset_subset = dataset_subset.shuffle(seed=1234).select(
            list(range(number_of_samples))
        )
        dataset_list.append(dataset_subset)

    dataset = concatenate_datasets(dataset_list)
    dataset = dataset.train_test_split(test_size=0.2)
    return dataset


def setup_special_tokens(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    special_tokens: List[str],
):
    # https://github.com/huggingface/trl/blob/66078c7c0142c7aada994856151e7e22745d4ecf/trl/models/utils.py#L43
    # We won't be changing bos, eos and pad token, though.
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # resize embedding layer to a multiple of 64, https://x.com/karpathy/status/1621578354024677377
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    return model, tokenizer


def main(training_args, det_callback, hparams):
    dataset = load_sft_dataset(hparams)

    model_name = hparams["model"]
    model, tokenizer = get_model_and_tokenizer(model_name, hparams["lora"])

    if hparams["chat_tokens"]["add_chat_tokens"]:
        model, tokenizer = setup_special_tokens(model, tokenizer, hparams["chat_tokens"]["special_tokens"])

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example["prompt"])):
            prompt = [
                {"role": "user", "content": example["prompt"][i]},
                {"role": "assistant", "content": example["text"][i]},
            ]
            text = tokenizer.apply_chat_template(prompt, tokenize=False)
            output_texts.append(text)
        return output_texts

    if hparams["data_collator"]["on_completions_only"]:
        assistant_prompt = hparams["data_collator"]["response_template"]
        response_template_ids = tokenizer.encode(
            assistant_prompt, add_special_tokens=False
        )
        collator = DataCollatorForCompletionOnlyLM(
            response_template_ids, tokenizer=tokenizer
        )
        logging.info("Using DataCollatorForCompletionOnlyLM.")
    else:
        collator = None
        logging.info("Using default data collator")

    trainer = SFTTrainer(
        args=training_args,
        model=model,
        tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        formatting_func=formatting_prompts_func,
        max_seq_length=8192,
    )

    trainer.add_callback(det_callback)

    trainer.train()


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        format=det.LOG_FORMAT, handlers=[logging.StreamHandler(sys.stdout)]
    )
    log_level = logging.INFO
    transformers.utils.logging.set_verbosity_info()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    info = det.get_cluster_info()
    hparams = info.trial.hparams
    training_args = TrainingArguments(**hparams["training_args"])
    if training_args.deepspeed:
        distributed = det.core.DistributedContext.from_deepspeed()
    else:
        distributed = det.core.DistributedContext.from_torch_distributed()

    with det.core.init(distributed=distributed) as core_context:
        det_callback = DetCallback(
            core_context,
            training_args,
        )
        main(training_args, det_callback, hparams)
