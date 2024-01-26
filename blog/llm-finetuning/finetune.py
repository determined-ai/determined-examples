import determined as det
import evaluate
from determined.transformers import DetCallback
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM

from dataset_utils import load_or_create_dataset


def get_chat_format(element):
    system_prompt = (
        "You are a helpful programmer assistant that excels at SQL. "
        "When prompted with a task and a definition of an SQL table, you "
        "respond with a SQL query to retrieve information from the table. "
        "Don't explain your reasoning, only provide the SQL query."
    )
    user_prompt = "Task: {instruction}\nSQL table: {input}\nSQL query: "

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.format_map(element)},
        {"role": "assistant", "content": element["response"]},
    ]


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


def main(training_args, det_callback, hparams):
    model_name = hparams["model"]
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(element):
        formatted = tokenizer.apply_chat_template(
            get_chat_format(element), tokenize=False
        )
        outputs = tokenizer(formatted)
        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }

    dataset = load_or_create_dataset(hparams["dataset_subset"])
    for k in dataset.keys():
        dataset[k] = dataset[k].map(tokenize)

    response_template_with_context = "\nSQL query: [/INST]"
    response_template_ids = tokenizer.encode(
        response_template_with_context, add_special_tokens=False
    )[1:]
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
    trainer.evaluate()
    trainer.train()


if __name__ == "__main__":
    info = det.get_cluster_info()
    hparams = info.trial.hparams
    distributed = det.core.DistributedContext.from_torch_distributed()
    with det.core.init(distributed=distributed) as core_context:
        training_args = TrainingArguments(**hparams["training_args"])
        det_callback = DetCallback(core_context, training_args)
        main(training_args, det_callback, hparams)
