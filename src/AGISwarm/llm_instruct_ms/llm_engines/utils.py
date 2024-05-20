
from typing import cast


def prepare_prompt(tokenizer, messages: list[dict], reply_prefix: str | None = None, tokenize: bool = False):
    """Prepare prompt for model"""
    if reply_prefix == "":
        reply_prefix = None
    prompt = cast(
        str,
        tokenizer.apply_chat_template(  # type: ignore
            messages,
            tokenize=tokenize,
            add_generation_prompt=reply_prefix is None,
        ),
    ) + ("$" if reply_prefix else "")
    if reply_prefix:
        prompt = prompt.replace("<|eot_id|>$", "<|eot_id|>assistant\n\n" + reply_prefix)

    return prompt
