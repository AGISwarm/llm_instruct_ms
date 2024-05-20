""" LLM Instruct Model Inference """

from threading import Thread
from typing import cast

import torch
import transformers

from .utils import prepare_prompt

SUPPORTED_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "unsloth/llama-3-8b-Instruct",
    "DevsDoCode/LLama-3-8b-Uncensored",
    "DevsDoCode/LLama-3-8b-Uncensored-4bit",
    "IlyaGusev/saiga_llama3_8b",
    "RichardErkhov/IlyaGusev_-_saiga_llama3_8b-4bits",
    "RLHFlow/LLaMA3-iterative-DPO-final",
]


MODEL_IS_4bit = {
    "meta-llama/Meta-Llama-3-8B-Instruct": False,
    "unsloth/llama-3-8b-Instruct-bnb-4bit": True,
    "unsloth/llama-3-8b-Instruct": False,
    "DevsDoCode/LLama-3-8b-Uncensored": False,
    "DevsDoCode/LLama-3-8b-Uncensored-4bit": True,
    "IlyaGusev/saiga_llama3_8b": False,
    "RichardErkhov/IlyaGusev_-_saiga_llama3_8b-4bits": True,
    "RLHFlow/LLaMA3-iterative-DPO-final": False,
    "apple/OpenELM-1_1B-Instruct": False,
}

BNB_CONFIG = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


class HFEngine:  # pylint: disable=invalid-name
    """LLM Instruct Model Inference"""

    def __init__(
        self,
        model_name: str = "IlyaGusev/saiga_llama3_8b",
        tokenizer_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    ):

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
        self.pipeline = cast(
            transformers.TextGenerationPipeline,
            transformers.pipeline(
                "text-generation",
                model=model_name,
                device_map="auto",
                tokenizer=self.tokenizer,
                model_kwargs={
                    "quantization_config": (
                        BNB_CONFIG if MODEL_IS_4bit[model_name] else None
                    )
                },
            ),
        )
        self.streamer = transformers.TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True  # type: ignore
        )

    async def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 100,
        reply_prefix: str = "",
        temperature: float = 0.5,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ):
        """Generate text from prompt"""
        prompt = prepare_prompt(self.tokenizer, messages, reply_prefix)
        print(f"Prompt: {prompt}")
        thread = Thread(
            target=self.pipeline,
            kwargs=dict(
                text_inputs=prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                streamer=self.streamer,
                clean_up_tokenization_spaces=True,
                repetition_penalty=repetition_penalty,
                # presence_penalty=presence_penalty,
                # frequency_penalty=frequency_penalty,
            ),
        )
        thread.start()
        for new_text in self.streamer:
            yield new_text
