""" LLM Instruct Model Inference """

import asyncio
from functools import partial
from threading import Thread
from typing import cast

import torch
import transformers
from pydantic import Field

from .utils import (
    EngineProtocol,
    SamplingParams,
    abort_generation_request,
    generation_request_queued_func,
    prepare_prompt,
)

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


class HFEngine(EngineProtocol[HFSamplingParams]):  # pylint: disable=invalid-name
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
        self.current_requests = {}

    @partial(generation_request_queued_func, wait_time=0.001)
    async def generate(
        self,
        request_id: str,
        messages: list[dict],
        reply_prefix: str = "",
        sampling_params: HFSamplingParams = HFSamplingParams(),
    ):
        """Generate text from prompt"""
        streamer = transformers.TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True  # type: ignore
        )
        prompt = prepare_prompt(self.tokenizer, messages, reply_prefix)
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
        yield {"request_id": request_id, "response": "success", "msg": reply_prefix}
        for new_text in streamer:
            await asyncio.sleep(0.001)
            yield {"request_id": request_id, "response": "success", "msg": new_text}

    async def abort(self, request_id: str):
        """Abort generation"""
        abort_generation_request(request_id)