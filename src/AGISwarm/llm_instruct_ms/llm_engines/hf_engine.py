""" LLM Instruct Model Inference """

from threading import Thread
from typing import Dict, List, cast

import torch
import transformers  # type: ignore
from pydantic import Field
from transformers import AutoTokenizer  # type: ignore

from .engine import Engine, SamplingParams

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


class HFSamplingParams(SamplingParams):
    """HF sampling settings"""

    repetition_penalty: float = Field(default=1.2, description="Repetition penalty")


# pylint: disable=too-few-public-methods
class HFEngine(Engine[HFSamplingParams]):  # pylint: disable=invalid-name
    """LLM Instruct Model Inference"""

    def __init__(
        self,
        hf_model_name: str,
        tokenizer_name: str | None,
    ):

        self.conversations: Dict[str, List[Dict]] = {}
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or hf_model_name)

        self.pipeline = cast(
            transformers.TextGenerationPipeline,
            transformers.pipeline(
                task="text-generation",
                model=hf_model_name,
                device_map="auto",
                tokenizer=self.tokenizer,
                model_kwargs={
                    "quantization_config": (
                        BNB_CONFIG if MODEL_IS_4bit[hf_model_name] else None
                    )
                },
            ),
        )

    async def generate(
        self,
        messages: list[dict],
        reply_prefix: str = "",
        sampling_params: HFSamplingParams = HFSamplingParams(),
    ):
        """Generate text from prompt"""
        streamer = transformers.TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True  # type: ignore
        )
        prompt = self.prepare_prompt(self.tokenizer, messages, reply_prefix)
        thread = Thread(
            target=self.pipeline,
            kwargs={
                "text_inputs": prompt,
                "do_sample": True,
                "streamer": streamer,
                "clean_up_tokenization_spaces": True,
            }
            | sampling_params.model_dump(),
        )
        thread.start()
        if reply_prefix:
            yield reply_prefix
        for new_text in streamer:
            yield cast(str, new_text)
