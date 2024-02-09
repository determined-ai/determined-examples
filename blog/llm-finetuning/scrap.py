from finetune import get_model_and_tokenizer
from dataset_utils import load_or_create_dataset
from chat_format import get_chat_format, get_response_template_ids
import tqdm

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
model, tokenizer = get_model_and_tokenizer(model_name)
max_length = 4096

def to_str(ids):
    return ",".join([str(i) for i in ids]) 

for difficulty in ["easy", "medium", "hard"]:
    dataset = load_or_create_dataset(difficulty)
    for element in tqdm.tqdm(dataset["train"]):
        element = tokenizer.apply_chat_template(get_chat_format(element, model_name), tokenize=False)
        element = tokenizer(element, padding=True, truncation=True, max_length=max_length, return_length=True)
        element, length = to_str(element["input_ids"]), element["length"][0]
        response_template = to_str(get_response_template_ids(tokenizer, model_name))
        if response_template not in element:
            raise ValueError("response_template not in element")
        if length >= max_length:
            raise ValueError("length >= max_length")