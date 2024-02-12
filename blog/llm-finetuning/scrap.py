import tqdm

from chat_format import get_chat_format, get_response_template_ids
from dataset_utils import load_or_create_dataset
from finetune import get_model_and_tokenizer

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v0.4"
model, tokenizer = get_model_and_tokenizer(model_name)


def validate():
    def to_str(ids):
        return ",".join([str(i) for i in ids])

    for difficulty in ["easy", "medium", "hard"]:
        dataset = load_or_create_dataset(difficulty)
        for split in ["train", "valid", "test"]:
            print(difficulty, split)
            for x in tqdm.tqdm(dataset[split]):
                formatted = tokenizer.apply_chat_template(
                    get_chat_format(x, model_name), tokenize=False
                )
                element = tokenizer(
                    formatted,
                    padding="longest",
                    truncation=True,
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


validate()
