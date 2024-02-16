from collections import defaultdict
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

from chat_format import get_chat_format, get_response_template_ids
from dataset_utils import load_or_create_dataset
from finetune import get_tokenize_fn, get_tokenizer

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = get_tokenizer(model_name)
tokenize_fn = get_tokenize_fn(tokenizer)
num_missing_response_template = defaultdict(lambda: defaultdict(int))
num_incomplete = defaultdict(lambda: defaultdict(int))
num_tokens = defaultdict(lambda: defaultdict(list))


def to_str(ids):
    return ",".join([str(i) for i in ids])


def plot_histogram_and_zoom_in(
    data,
    bins,
    filename_prefix,
):
    # Step 1: Plot the original histogram
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
    plt.title(f"{filename_prefix} Token Histogram")
    plt.savefig(f"{filename_prefix}.png")
    plt.close()


def get_collate_fn(difficulty, split):
    def fn(x):
        formatted = []
        for e in x:
            formatted.append(
                tokenizer.apply_chat_template(
                    get_chat_format(e, model_name), tokenize=False
                )
            )
        untruncated = tokenizer(formatted, padding=False, truncation=False)["input_ids"]
        element = tokenize_fn(formatted)["input_ids"]
        response_template = to_str(get_response_template_ids(tokenizer, model_name))
        for i, e in enumerate(element):
            num_tokens[difficulty][split].append(len(untruncated[i]))
            if response_template not in to_str(e):
                num_missing_response_template[difficulty][split] += 1
            decoded = tokenizer.decode(e)
            if x[i]["response"] not in decoded:
                num_incomplete[difficulty][split] += 1

        return element

    return fn


def validate():
    for difficulty in ["easy", "medium", "hard"]:
        dataset = load_or_create_dataset(difficulty)
        for split in ["train", "valid", "test"]:
            print(difficulty, split)
            dataloader = torch.utils.data.DataLoader(
                dataset[split],
                batch_size=1,
                collate_fn=get_collate_fn(difficulty, split),
            )
            for _ in tqdm.tqdm(dataloader):
                pass

            plot_histogram_and_zoom_in(
                np.array(num_tokens[difficulty][split]),
                bins=100,
                filename_prefix=f"{difficulty}_{split}_tokens",
            )

    pprint(num_missing_response_template)
    pprint(num_incomplete)


validate()
