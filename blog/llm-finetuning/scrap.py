from collections import defaultdict
from pprint import pprint

import torch
import tqdm

from chat_format import get_chat_format, get_response_template_ids
from dataset_utils import load_or_create_dataset
from finetune import get_model_and_tokenizer, get_tokenize_fn

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
model, tokenizer = get_model_and_tokenizer(model_name)
tokenize_fn = get_tokenize_fn(tokenizer)
num_missing_response_template = defaultdict(lambda: defaultdict(int))
num_incomplete = defaultdict(lambda: defaultdict(int))


def to_str(ids):
    return ",".join([str(i) for i in ids])


def get_collate_fn(difficulty, split):
    def fn(x):
        formatted = []
        for e in x:
            formatted.append(
                tokenizer.apply_chat_template(
                    get_chat_format(e, model_name), tokenize=False
                )
            )
        element = tokenize_fn(formatted)["input_ids"]
        response_template = to_str(get_response_template_ids(tokenizer, model_name))
        for i, e in enumerate(element):
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
                batch_size=2,
                collate_fn=get_collate_fn(difficulty, split),
            )
            for _ in tqdm.tqdm(dataloader):
                pass

    pprint(num_missing_response_template)
    pprint(num_incomplete)


validate()
