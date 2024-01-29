import argparse
import glob

from determined.experimental import client

from chat_format import ASSISTANT_PROMPT, EOS_TOKEN, get_chat_format
from dataset_utils import load_or_create_dataset
from finetune import get_model_and_tokenizer


def main(exp_id, dataset_subset):
    if exp_id is None:
        checkpoint_dir = "TinyLlama/TinyLlama-1.1B-Chat-v0.4"
    else:
        exp = client.get_experiment(exp_id)
        checkpoint = exp.list_checkpoints(
            max_results=1,
            sort_by=client.CheckpointSortBy.SEARCHER_METRIC,
            order_by=client.OrderBy.DESCENDING,
        )[0]
        checkpoint_dir = checkpoint.download(mode=client.DownloadMode.MASTER)
        checkpoint_dir = glob.glob(f"{checkpoint_dir}/checkpoint-*")[0]

    model, tokenizer = get_model_and_tokenizer(checkpoint_dir)
    eos_token_id = tokenizer.get_vocab()[EOS_TOKEN]

    dataset = load_or_create_dataset(dataset_subset)["test"]
    element = dataset[0]
    formatted = tokenizer.apply_chat_template(
        get_chat_format(element)[:2],
        tokenize=False,
    )
    formatted += ASSISTANT_PROMPT
    print(formatted)

    inputs = tokenizer(formatted, return_tensors="pt")
    outputs = model.generate(**inputs, eos_token_id=eos_token_id, max_new_tokens=1000)
    input_length = inputs["input_ids"].shape[1]
    response = tokenizer.batch_decode(
        outputs[:, input_length:], skip_special_tokens=True
    )
    print(f"\n\nCorrect response:\n{element['response']}")
    print(f"\n\nLLM response:\n{response[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", type=int, default=None, required=False)
    parser.add_argument("--dataset_subset", type=str, default="easy", required=False)
    args = parser.parse_args()
    main(args.exp_id, args.dataset_subset)
