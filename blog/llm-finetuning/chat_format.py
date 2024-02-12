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

CHAT_ML_ASSISTANT_PROMPT = "<|im_start|>assistant\n"

CHAT_ML_EOS_TOKEN = "<|im_end|>"


def get_chat_format(element, model_name):
    user_prompt = "Task: {instruction}\nSQL table: {input}\nSQL query: "
    output = [
        {"role": "user", "content": user_prompt.format_map(element)},
        {"role": "assistant", "content": element["response"]},
    ]

    if model_name != "mistralai/Mistral-7B-Instruct-v0.2":
        system_prompt = (
            "You are a helpful programmer assistant that excels at SQL. "
            "When prompted with a task and a definition of an SQL table, you "
            "respond with a SQL query to retrieve information from the table. "
            "Don't explain your reasoning, only provide the SQL query."
        )
        output = [{"role": "system", "content": system_prompt}] + output

    return output


def set_special_tokens(tokenizer, model_name):
    if model_name == "TinyLlama/TinyLlama-1.1B-Chat-v0.4":
        tokenizer.chat_template = CHAT_ML_TEMPLATE
        tokenizer.eos_token = CHAT_ML_EOS_TOKEN
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id


def get_response_template_ids(tokenizer, model_name):
    if model_name == "TinyLlama/TinyLlama-1.1B-Chat-v0.4":
        return tokenizer.encode(CHAT_ML_ASSISTANT_PROMPT, add_special_tokens=False)
    else:
        return tokenizer.encode("[/INST]", add_special_tokens=False)
