from finetune import get_model_and_tokenizer
from dataset_utils import load_or_create_dataset
from chat_format import (
    get_chat_format,
    get_response_template_ids,
    get_tokenizer_max_length,
)
import tqdm

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
model, tokenizer = get_model_and_tokenizer(model_name)


def validate():
    def to_str(ids):
        return ",".join([str(i) for i in ids])

    for difficulty in ["easy", "medium", "hard"]:
        dataset = load_or_create_dataset(difficulty)
        max_length = get_tokenizer_max_length(difficulty, model_name)
        for split in ["train", "valid", "test"]:
            print(difficulty, split)
            for x in tqdm.tqdm(dataset[split]):
                formatted = tokenizer.apply_chat_template(
                    get_chat_format(x, model_name), tokenize=False
                )
                element = tokenizer(
                    formatted, padding=True, truncation=True, max_length=max_length
                )["input_ids"]
                response_template = to_str(
                    get_response_template_ids(tokenizer, model_name)
                )
                if response_template not in to_str(element):
                    print(formatted)
                    raise ValueError("response_template not in element")
                # decoded = tokenizer.decode(element)
                # if decoded != formatted:
                #     print(f"decoded {decoded}")
                #     print(f"formatted {formatted}")
                #     raise ValueError("decoded != formatted")


def get_max_length():
    for difficulty in ["easy", "medium", "hard"]:
        max_length = 0
        dataset = load_or_create_dataset(difficulty)
        for split in ["train", "valid", "test"]:
            print(difficulty, split)
            for element in tqdm.tqdm(dataset[split]):
                element = tokenizer.apply_chat_template(
                    get_chat_format(element, model_name), tokenize=False
                )
                element = tokenizer(element, return_length=True)
                max_length = max(max_length, element["length"][0])
        print("max_length", max_length)


# get_max_length()
validate()
