import logging
import os
import sys
from typing import Any, Dict, List

import datasets
import determined as det
import torch
import transformers
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from determined.transformers import DetCallback
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import DPOTrainer

from chat_format import CHAT_ML_TEMPLATE, get_response_template_ids

logger = logging.getLogger(__name__)

TRIPLET_DATASET = "prompt,chosen,rejected"
CONVERSATION_DATASET = "chosen,rejected"


def get_tokenizer(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        # Following DPO alignment handbook
        truncation_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 8192

    if tokenizer.chat_template is None and tokenizer.default_chat_template is None:
        tokenizer.chat_template = CHAT_ML_TEMPLATE

    tokenizer.add_eos_token = False
    return tokenizer


def get_model(model_name_or_path, use_lora, inference=False, device_map="auto"):
    if inference:
        if use_lora:
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_name_or_path, torch_dtype=torch.bfloat16, device_map=device_map
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
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

    return model


def is_feature_chat_conversation_format(dataset: Dataset, feature: str) -> bool:
    example = dataset[0][feature]
    if isinstance(example, list) and all(isinstance(x, dict) for x in example):
        for sample in example:
            if "content" not in sample or "role" not in sample:
                raise RuntimeError(
                    f"Column {feature} has data in unsupported format : {sample}"
                )
        return True
    else:
        raise RuntimeError(
            f"Column {feature} has data in unsupported format : {example}"
        )


def get_dataset_format(dataset: Dataset) -> str:
    if "chosen" not in dataset.features or "rejected" not in dataset.features:
        raise RuntimeError(
            f"DPO-compatible dataset requires 'chosen' and 'rejected' features."
        )

    if all(feature in dataset.features for feature in ["prompt", "chosen", "rejected"]):
        return TRIPLET_DATASET

    if is_feature_chat_conversation_format(
        dataset, "chosen"
    ) and is_feature_chat_conversation_format(dataset, "rejected"):
        return CONVERSATION_DATASET


def process_conversation_dataset(dataset: Dataset, tokenizer) -> Dataset:
    processed_data = {"prompt": [], "chosen": [], "rejected": []}

    for example in dataset:
        assert ". ".join([x["content"] for x in example["chosen"][:-1]]) == ". ".join(
            [x["content"] for x in example["rejected"][:-1]]
        )
        assert all(x["role"] != "system" for x in example["chosen"])

        prompt_messages = example["chosen"][:-1]
        chosen_messages = example["chosen"][-1:]
        rejected_messages = example["rejected"][-1:]

        processed_data["prompt"].append(
            tokenizer.apply_chat_template(prompt_messages, tokenize=False)
        )
        processed_data["chosen"].append(
            tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        )
        processed_data["rejected"].append(
            tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        )

    dataset = Dataset.from_dict(processed_data)
    return dataset


def process_triplet_dataset(
    dataset: Dataset, tokenizer: PreTrainedTokenizer
) -> Dataset:
    def apply_chat_template(example):
        if "system" in example:
            prompt = example["system"] + "\n"
        else:
            prompt = ""

        example["prompt"] = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt + example["prompt"]}],
            tokenize=False,
        )
        example["chosen"] = tokenizer.apply_chat_template(
            [{"role": "assistant", "content": example["chosen"]}], tokenize=False
        )
        example["rejected"] = tokenizer.apply_chat_template(
            [{"role": "assistant", "content": example["rejected"]}], tokenize=False
        )
        return example

    columns = set(dataset.features) - {"prompt", "rejected", "chosen"}
    dataset = dataset.map(apply_chat_template, remove_columns=list(columns))
    return dataset


def load_dpo_datasets(
    datasets: List[str], tokenizer: PreTrainedTokenizer
) -> DatasetDict:
    dataset_list_validated = []
    for dataset_name in datasets:
        dataset = load_dataset(dataset_name)
        if isinstance(dataset, DatasetDict):
            dataset_list = [dataset[k] for k in dataset]
        else:
            dataset_list = [dataset]

        for ds in dataset_list:
            dataset_format = get_dataset_format(ds)
            if dataset_format == CONVERSATION_DATASET:
                ds = process_conversation_dataset(ds, tokenizer)
            elif dataset_format == TRIPLET_DATASET:
                ds = process_triplet_dataset(ds, tokenizer)

            dataset_list_validated.append(ds)

    dataset = concatenate_datasets(dataset_list_validated)
    dataset = dataset.train_test_split(test_size=0.2)
    return dataset


def download_ckpt(ckpt_uuid: str, core_context: det.core.Context) -> str:
    download_dir = os.path.join(os.environ.get("HF_CACHE", "."), ckpt_uuid)

    if not os.path.exists(os.path.join(download_dir, ckpt_uuid)):
        os.makedirs(download_dir)

        def selector(path: str) -> bool:
            if any(
                [
                    path.endswith(ext)
                    for ext in [
                        "config.json",
                        "generation-config.json",
                        ".safetensors",
                        "special_tokens_map.json",
                        "tokenizer_config.json",
                        "tokenizer.json",
                        "tokenizer.model",
                    ]
                ]
            ):
                return True

            return False

        core_context.checkpoint.download(ckpt_uuid, download_dir, selector=selector)

    model_dir = get_last_checkpoint(download_dir)
    return model_dir


def main(
    core_context: det.core.Context,
    training_args: TrainingArguments,
    det_callback: DetCallback,
    hparams: Dict[str, Any],
) -> None:

    model_ckpt = hparams.get("model_ckpt", None)
    if model_ckpt:
        model_name_or_path = download_ckpt(model_ckpt, core_context)
    else:
        model_name_or_path = hparams["model_name"]

    model = get_model(model_name_or_path, hparams["lora"])
    model_ref = get_model(model_name_or_path, hparams["lora"])
    tokenizer = get_tokenizer(model_name_or_path)

    dataset = load_dpo_datasets(hparams["datasets"], tokenizer)

    trainer = DPOTrainer(
        model,
        ref_model=model_ref,
        args=training_args,
        beta=hparams["dpo_beta"],
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        loss_type=hparams["dpo_loss"],
        tokenizer=tokenizer,
        max_length=8192,
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

    with det.core.init() as core_context:
        det_callback = DetCallback(
            core_context,
            training_args,
        )
        main(core_context, training_args, det_callback, hparams)
