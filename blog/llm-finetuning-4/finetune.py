import logging
import sys
from transformers import Trainer, default_data_collator
import datasets
import determined as det
import evaluate
import torch
import transformers
import wandb
from determined.transformers import DetCallback
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM
from itertools import chain
from typing import Dict

from chat_format import get_chat_format, get_response_template_ids, set_special_tokens
from dataset_utils import load_or_create_dataset

logger = logging.getLogger(__name__)

block_size = 1024

def get_tokenizer(model_name, model_commit_hash):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="right",
        truncation_side="right",
        revision=model_commit_hash,
    )
    set_special_tokens(tokenizer, model_name)
    return tokenizer


def get_model_and_tokenizer(model_name, use_lora, hparams, inference=False, device_map="auto", model_commit_hash=None):
    if inference:
        if use_lora:
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map=device_map, revision=model_commit_hash
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                revision=model_commit_hash,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            revision=model_commit_hash,
        )
        # model.gradient_checkpointing_enable({"use_reentrant": False})
        if use_lora:
            r = hparams["r"]
            lora_alpha = r * hparams["lora_alpha_in_r"]
            peft_config = LoraConfig(
                task_type="CAUSAL_LM",
                inference_mode=False,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=hparams["lora_dropout"],
            )

            model = get_peft_model(model, peft_config)
            

    tokenizer = get_tokenizer(model_name, model_commit_hash=model_commit_hash)
    return model, tokenizer


def get_tokenize_fn(tokenizer):
    def fn(formatted):
        return tokenizer(formatted, padding=True, truncation=True, max_length=2048)

    return fn


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


def group_texts(examples) -> Dict:
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead
    # of this drop, you can customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def main(training_args, det_callback, hparams):
    if "hf_token" in hparams:
        import huggingface_hub

        huggingface_hub.login(token=hparams["hf_token"])
    
    model_name = hparams["model"]
    model_commit_hash = None
    if "model_commit_hash" in hparams:
        model_commit_hash = hparams["model_commit_hash"]
    model, tokenizer = get_model_and_tokenizer(model_name, hparams["lora"], hparams=hparams, model_commit_hash=model_commit_hash)
    tokenize_fn = get_tokenize_fn(tokenizer)

    def tokenize(element):
        formatted = tokenizer.apply_chat_template(
            get_chat_format(element, model_name), tokenize=False
        )
        outputs = tokenize_fn(formatted)
        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }

    dataset = load_or_create_dataset(hparams["dataset_subset"])
    column_names = list(dataset["train"].features)
    for k in dataset.keys():
        dataset[k] = dataset[k].map(tokenize, remove_columns=column_names)

    response_template_ids = get_response_template_ids(tokenizer, model_name)
    collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer
    )

    with training_args.main_process_first(desc="grouping texts together", local=False):
        lm_datasets = dataset.map(
            group_texts,
            batched=True,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    bleu = evaluate.load("bleu")
    acc = evaluate.load("accuracy")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        labels = labels[:, 1:]
        preds = preds[:, :-1]
        # -100 is a default value for ignore_index used by DataCollatorForCompletionOnlyLM
        mask = labels == -100
        labels[mask] = tokenizer.pad_token_id
        preds[mask] = tokenizer.pad_token_id

        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels)
        accuracy = acc.compute(predictions=preds[~mask], references=labels[~mask])

        return {**bleu_score, **accuracy}

    trainer = Trainer(
        args=training_args,
        model=model,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["valid"],
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
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
    import os

    # Get the current working directory
    current_directory = os.getcwd()

    if hparams["training_args"]["deepspeed"]:
        hparams["training_args"]["deepspeed"] = "ds_configs/ds_config_stage_3.json"
    training_args = TrainingArguments(**hparams["training_args"])

    if training_args.deepspeed:
        os.environ["LOCAL_SIZE"] = os.environ["LOCAL_WORLD_SIZE"]
        os.environ["CROSS_RANK"] = str(int(os.environ["RANK"]) // int(os.environ["LOCAL_WORLD_SIZE"]))
        os.environ["CROSS_SIZE"] = str(int(os.environ["WORLD_SIZE"]) // int(os.environ["LOCAL_WORLD_SIZE"]))
        os.environ["CHIEF_IP"] = os.environ["DET_CHIEF_IP"]
        distributed = det.core.DistributedContext.from_deepspeed()
    else:
        distributed = det.core.DistributedContext.from_torch_distributed()

    with det.core.init(distributed=distributed) as core_context:
        if core_context.distributed.rank == 0:
            wandb.login(key=hparams["wandb_key"])
            import uuid
            # Generate a UUID
            my_uuid = uuid.uuid4()
            # Convert UUID to string
            uuid_str = str(my_uuid)[:5]
            r = hparams["r"]
            lora_alpha = hparams["lora_alpha_in_r"]
            lora_dropout = hparams["lora_dropout"]
            dataset_subset = hparams["dataset_subset"]
            run_name = f"test_lora_blog_{dataset_subset}_r_{r}_alpha_in_r_{lora_alpha}_dropout_{lora_dropout}_{uuid_str}"
            run = wandb.init(
                project="lora-blog", 
                name=run_name, 
                config={
                    "r":hparams["r"],
                    "lora_alpha_in_r":hparams["lora_alpha_in_r"],
                    "dropout":hparams["lora_dropout"],
                    "dataset_subset":hparams["dataset_subset"],
                    "model":hparams["model"]
                }
            )
        det_callback = DetCallback(
            core_context,
            training_args,
        )
        main(training_args, det_callback, hparams)
