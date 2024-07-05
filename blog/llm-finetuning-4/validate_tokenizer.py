from collections import defaultdict
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

from chat_format import get_assistant_prompt, get_chat_format, get_response_template_ids
from dataset_utils import load_or_create_dataset
from finetune import get_tokenize_fn, get_tokenizer

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = get_tokenizer(model_name)
tokenize_fn = get_tokenize_fn(tokenizer)
num_missing_response_template = defaultdict(lambda: defaultdict(int))
num_incomplete = defaultdict(lambda: defaultdict(int))
num_tokens = defaultdict(lambda: defaultdict(list))
num_tokens_before_response = defaultdict(lambda: defaultdict(list))
num_tokens_with_padding_and_truncation = defaultdict(lambda: defaultdict(list))


def to_str(ids):
    return ",".join([str(i) for i in ids])


def plot_histogram(
    data,
    bins,
    title,
    filename_prefix,
):
    hist_data, bin_edges = np.histogram(data, bins=bins)
    plt.figure()
    plt.bar(
        (bin_edges[:-1] + bin_edges[1:]) / 2,
        hist_data,
        width=np.diff(bin_edges),
        edgecolor="black",
    )
    plt.xlabel("Bin")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.savefig(f"{filename_prefix}.png")
    plt.close()

    return bin_edges


def get_collate_fn(difficulty, split):
    def fn(x):
        formatted = []
        before_response_formatted = []
        for e in x:
            with_chat_template = tokenizer.apply_chat_template(
                get_chat_format(e, model_name), tokenize=False
            )
            formatted.append(with_chat_template)
            before_response_formatted.append(
                with_chat_template.split(get_assistant_prompt(model_name))[0]
            )
        untruncated = tokenizer(formatted, padding=False, truncation=False)["input_ids"]
        before_response_untruncated = tokenizer(
            before_response_formatted,
            padding=False,
            truncation=False,
        )["input_ids"]
        element = tokenize_fn(formatted)["input_ids"]
        response_template = to_str(get_response_template_ids(tokenizer, model_name))
        for i, e in enumerate(element):
            num_tokens[difficulty][split].append(len(untruncated[i]))
            num_tokens_before_response[difficulty][split].append(
                len(before_response_untruncated[i])
            )
            num_tokens_with_padding_and_truncation[difficulty][split].append(len(e))
            if response_template not in to_str(e):
                num_missing_response_template[difficulty][split] += 1
            decoded = tokenizer.decode(e)
            if x[i]["response"] not in decoded:
                num_incomplete[difficulty][split] += 1

        return element

    return fn


def validate():
    batch_size = 4
    for difficulty in ["easy", "medium", "hard"]:
        dataset = load_or_create_dataset(difficulty)
        for split in ["train", "valid", "test"]:
            print(difficulty, split)
            dataloader = torch.utils.data.DataLoader(
                dataset[split],
                batch_size=batch_size,
                collate_fn=get_collate_fn(difficulty, split),
            )
            for _ in tqdm.tqdm(dataloader):
                pass

            plot_histogram(
                np.array(num_tokens[difficulty][split]),
                bins=100,
                title=f"{difficulty} {split} # Tokens",
                filename_prefix=f"{difficulty}_{split}_tokens",
            )

            plot_histogram(
                np.array(num_tokens_before_response[difficulty][split]),
                bins=100,
                title=f"{difficulty} {split} # Tokens Before Response",
                filename_prefix=f"{difficulty}_{split}_tokens_before_response",
            )

            plot_histogram(
                np.array(num_tokens_with_padding_and_truncation[difficulty][split]),
                bins=100,
                title=f"{difficulty} {split} # Tokens with Padding & Truncation",
                filename_prefix=f"{difficulty}_{split}_tokens_with_pad_trunc_{batch_size}",
            )

    pprint(num_missing_response_template)
    pprint(num_incomplete)


validate()
