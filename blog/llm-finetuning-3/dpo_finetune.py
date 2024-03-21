import logging
import random
import sys
from typing import Any, Dict, List

import datasets
import determined as det
import transformers
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from determined.transformers import DetCallback
from transformers import PreTrainedTokenizer, TrainingArguments
from trl import DPOTrainer

from utils import download_ckpt, get_model, get_tokenizer

logger = logging.getLogger(__name__)

TRIPLET_DATASET = "prompt,chosen,rejected"
CONVERSATION_DATASET = "chosen,rejected"


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

    model = get_model(model_name_or_path)
    if not hparams["precompute_ref_log_probs"]:
        model_ref = get_model(model_name_or_path)
        model_ref.eval()
    else:
        model_ref = None

    tokenizer = get_tokenizer(
        model_name_or_path,
        truncation_side="left",
        model_max_length=hparams["max_length"],
        add_eos_token=False,
    )
    dataset = load_dpo_datasets(hparams["datasets"], tokenizer)

    if core_context.distributed.rank == 0:
        for index in random.sample(range(len(dataset["train"])), 3):
            logger.info(
                f"Prompt sample {index} of the raw training set:\n\n{dataset['train'][index]['prompt']}"
            )
            logger.info(
                f"Chosen sample {index} of the raw training set:\n\n{dataset['train'][index]['chosen']}"
            )
            logger.info(
                f"Rejected sample {index} of the raw training set:\n\n{dataset['train'][index]['rejected']}"
            )

    trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=hparams["dpo_beta"],
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        loss_type=hparams["dpo_loss"],
        tokenizer=tokenizer,
        precompute_ref_log_probs=hparams["precompute_ref_log_probs"],
        max_length=hparams["max_length"],
        max_prompt_length=hparams["max_prompt_length"],
        max_target_length=hparams["max_target_length"],
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

    distributed = det.core.DistributedContext.from_deepspeed()
    with det.core.init(distributed=distributed) as core_context:
        det_callback = DetCallback(
            core_context,
            training_args,
        )
        main(core_context, training_args, det_callback, hparams)
