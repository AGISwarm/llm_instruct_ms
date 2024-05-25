"""LLaMA C++ Engine"""

import asyncio
from functools import partial
from typing import cast

from llama_cpp import CreateCompletionStreamResponse, Llama
from pydantic import Field
from transformers import AutoTokenizer

from .utils import (
    EngineProtocol,
    SamplingParams,
    abort_generation_request,
    generation_request_queued_func,
    prepare_prompt,
)

__SUPPORTED_MODELS = [
    "MaziyarPanahi/Meta-Llama-3-70B-Instruct-GGUF",
    "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    "Orenguteng/Llama-3-8B-Lexi-Uncensored-GGUF",
]


class LlamaCppSamplingParams(SamplingParams):
    """LlamaCpp sampling settings"""

    repetition_penalty: float = Field(default=1.2, description="Repetition penalty")
    frequency_penalty: float = Field(default=0.0, description="Frequency penalty")
    presence_penalty: float = Field(default=0.0, description="Presence penalty")


class LlamaCppEngine(EngineProtocol[LlamaCppSamplingParams]):
    """LLM Instruct Model Inference"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        model_name: str,
        tokenizer_name: str | None,
        filename: str,
        n_gpu_layers: int = -1,
        n_ctx: int = 8192,
    ):
        self.llama = Llama.from_pretrained(
            model_name, filename=filename, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx
        )
        self.tokenizer: object = AutoTokenizer.from_pretrained(
            tokenizer_name or model_name
        )

    def get_sampling_params(self, sampling_params: LlamaCppSamplingParams):
        """Get sampling params"""
        sampling_params_dict = sampling_params.model_dump()
        sampling_params_dict["max_tokens"] = sampling_params_dict.pop("max_new_tokens")
        sampling_params_dict["repeat_penalty"] = sampling_params_dict.pop(
            "repetition_penalty"
        )
        return sampling_params_dict

    @partial(generation_request_queued_func, wait_time=0.001)
    async def generate(
        self,
        request_id: str,
        messages: list[dict],
        reply_prefix: str = "",
        sampling_params: LlamaCppSamplingParams = LlamaCppSamplingParams(),
    ):
        """Generate text from prompt"""
        prompt = prepare_prompt(self.tokenizer, messages, reply_prefix)
        if reply_prefix:
            yield {
                "request_id": request_id,
                "response": "success",
                "msg": reply_prefix,
            }
        sampling_params_dict = self.get_sampling_params(sampling_params)
        for output in self.llama(
            prompt,
            **sampling_params_dict,
            stream=True,
        ):
            output = cast(CreateCompletionStreamResponse, output)
            await asyncio.sleep(0.001)
            yield {
                "request_id": request_id,
                "response": "success",
                "msg": output["choices"][0]["text"],
            }

    async def abort(self, request_id: str):
        """Abort generation"""
        abort_generation_request(request_id)
