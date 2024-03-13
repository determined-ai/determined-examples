import logging
import sys

import datasets
import determined as det
import evaluate
import torch
import transformers
from determined.transformers import DetCallback
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer, setup_chat_format
from chat_format import get_chat_format, get_response_template_ids, CHAT_ML_TEMPLATE
from dataset_utils import load_or_create_dataset
from datasets import load_dataset, concatenate_datasets

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
        dataset_subset = load_dataset(dataset_name, subset_info["subset"])
        if "ratio" in subset_info:
            number_of_samples = int(len(dataset_subset)*subset_info["ratio"])
        elif "number_of_samples" in subset_info:
            number_of_samples = subset_info["number_of_samples"]
        else:
            raise RuntimeError(f"Unknown subset definition {subset_info}")
        dataset_subset = dataset_subsets.shuffle(seed=1234).select(list(range(number_of_samples)))
        dataset_list.append(dataset_subset)

    dataset = concatenate_datasets(dataset_list)
    dataset = dataset.train_test_split(test_size=0.2)
    return dataset


def main(training_args, det_callback, hparams):
    model_name = hparams["model"]
    model, tokenizer = get_model_and_tokenizer(model_name, hparams["lora"])

    dataset = load_sft_dataset(hparams)

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['prompt'])):
            prompt = [{"role": "user", "content": example["prompt"][i]},
                      {"role": "assistant", "content": example["text"][i]}]
            text = tokenizer.apply_chat_template(prompt, tokenize=False)
            output_texts.append(text)
        return output_texts

    # response_template_ids = get_response_template_ids(tokenizer)
    # collator = DataCollatorForCompletionOnlyLM(
    #     response_template_ids, tokenizer=tokenizer
    # )

    trainer = SFTTrainer(
        args=training_args,
        model=model,
        tokenizer=tokenizer,
        #data_collator=collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        formatting_func=formatting_prompts_func,
        max_seq_length=2048,
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
