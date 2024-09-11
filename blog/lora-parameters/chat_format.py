CHAT_ML_TEMPLATE = """
{% for message in messages %}
{% if message['role'] == 'user' %}
{{'<|im_start|>user\n' + message['content'].strip() + '<|im_end|>' }}
{% elif message['role'] == 'system' %}
{{'<|im_start|>system\n' + message['content'].strip() + '<|im_end|>' }}
{% elif message['role'] == 'assistant' %}
{{'<|im_start|>assistant\n'  + message['content'] + '<|im_end|>' }}
{% endif %}
{% endfor %}
"""


CHAT_ML_EOS_TOKEN = "<|im_end|>"


def get_chat_format(element, model_name, with_assistant_response=True):
    system_prompt = (
        "You are a helpful programmer assistant that excels at SQL. "
        "When prompted with a task and a definition of an SQL table, you "
        "respond with a SQL query to retrieve information from the table. "
        "Don't explain your reasoning, only provide the SQL query."
    )

    user_prompt = "Task: {instruction}\nSQL table: {input}\nSQL query: "

    if model_name == "mistralai/Mistral-7B-Instruct-v0.2":
        user_prompt = f"{system_prompt}\n{user_prompt}"
        output = [
            {"role": "user", "content": user_prompt.format_map(element)},
        ]
    else:
        output = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format_map(element)},
        ]

    if with_assistant_response:
        output.append({"role": "assistant", "content": element["response"]})

    return output


def set_special_tokens(tokenizer, model_name):
    if model_name == "TinyLlama/TinyLlama-1.1B-Chat-v0.4":
        tokenizer.chat_template = CHAT_ML_TEMPLATE
        tokenizer.eos_token = CHAT_ML_EOS_TOKEN
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id


def get_assistant_prompt(model_name):
    if model_name == "TinyLlama/TinyLlama-1.1B-Chat-v0.4":
        return "<|im_start|>assistant\n"
    else:
        return "[/INST]"


def get_response_template_ids(tokenizer, model_name):
    return tokenizer.encode(get_assistant_prompt(model_name), add_special_tokens=False)


def maybe_add_generation_prompt(x, model_name):
    if model_name == "TinyLlama/TinyLlama-1.1B-Chat-v0.4":
        return x + get_assistant_prompt(model_name)
    else:
        return x