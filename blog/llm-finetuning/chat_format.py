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

ASSISTANT_PROMPT = "<|im_start|>assistant\n"

EOS_TOKEN = "<|im_end|>"


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
