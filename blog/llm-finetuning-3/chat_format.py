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

CHAT_ML_END_TURN_TOKEN = "<|im_end|>"
CHAT_ML_START_TURN_TOKEN = "<|im_start|>"


def get_assistant_prompt():
    return "<|im_start|>assistant\n"


def get_response_template_ids(tokenizer):
    return tokenizer.encode(get_assistant_prompt(), add_special_tokens=False)


def maybe_add_generation_prompt(text: str) -> str:
    return text + get_assistant_prompt()
