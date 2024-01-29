import logging

import datasets
import determined as det
import evaluate
import transformers
from determined.transformers import DetCallback
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM
import sys
from chat_format import ASSISTANT_PROMPT, CHAT_ML_TEMPLATE, EOS_TOKEN, get_chat_format
from dataset_utils import load_or_create_dataset

logger = logging.getLogger(__name__)


def get_model_and_tokenizer(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token="<ADD_YOUR_TOKEN>",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_auth_token="<ADD_YOUR_TOKEN>",
        padding_side="left",
        truncation_side="right",
        add_eos_token=True,
    )
    if model_name == "TinyLlama/TinyLlama-1.1B-Chat-v0.4":
        tokenizer.chat_template = CHAT_ML_TEMPLATE
        tokenizer.eos_token = EOS_TOKEN
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


def main(training_args, det_callback, hparams):
    model_name = hparams["model"]
    model, tokenizer = get_model_and_tokenizer(model_name)

    def tokenize(element):
        formatted = tokenizer.apply_chat_template(
            get_chat_format(element), tokenize=False
        )
        outputs = tokenizer(formatted, padding=True, truncation=True, max_length=1024)
        logging.error(f"type(output_ids_={type(outputs['input_ids'])}")
        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }

    dataset = load_or_create_dataset(hparams["dataset_subset"])
    column_names = list(dataset["train"].features)
    for k in dataset.keys():
        dataset[k] = dataset[k].map(tokenize, remove_columns=column_names)

    if model_name == "TinyLlama/TinyLlama-1.1B-Chat-v0.4":
        response_template_ids = tokenizer.encode(
            ASSISTANT_PROMPT, add_special_tokens=False
        )
    else:
        response_template_ids = tokenizer.encode("[/INST]", add_special_tokens=False)
    collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer
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

    logging.error(f"dataset={dataset['train'][0]}")

    trainer = Trainer(
        args=training_args,
        model=model,
        tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
    )

    trainer.add_callback(det_callback)
    # we need to comment this one out, since it will lead to the following error:
    # [parameter_offload.py:86:_apply_to_tensors_only] A module has unknown inputs or outputs type (<class 'transformers.cache_utils.DynamicCache'>)
    # and the tensors embedded in it cannot be detected. The ZeRO-3 hooks designed to trigger before or after backward pass of the module relies on
    # knowing the input and output tensors and therefore may not get triggered properly.
    # The error happens due to deepspeed initialization happening in the trainer.train(), hence call on eval fails.

    # trainer.evaluate()

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
        det_callback = DetCallback(core_context, training_args)
        main(training_args, det_callback, hparams)
