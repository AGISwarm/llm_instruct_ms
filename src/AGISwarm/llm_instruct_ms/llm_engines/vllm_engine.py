""" LLM Instruct Model Inference """

import asyncio
import logging

import torch
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

from .utils import prepare_prompt


class VLLMEngine:
    """LLM Instruct Model Inference using VLLM"""

    def __init__(self, model_name: str):
        self.model = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(
                model=model_name,
                tokenizer=model_name,
                dtype=torch.float16,
                tensor_parallel_size=2,
                gpu_memory_utilization=1.0,
                trust_remote_code=True,
            )
        )
        logging.info("Model loaded")
        self.tokenizer = asyncio.run(self.model.get_tokenizer())

    async def generate(  # type: ignore
        self,
        request_id: str,
        messages: list[dict],
        max_new_tokens: int = 100,
        reply_prefix: str | None = None,
        temperature: float = 0.5,
        top_p: float = 0.95,
        repetition_penalty: float = 1,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
    ):
        """Generate text from prompt"""
        prompt = prepare_prompt(self.tokenizer, messages, reply_prefix)
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            skip_special_tokens=True,
            truncate_prompt_tokens=True,
            stop_token_ids=[
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ],
        )
        if reply_prefix:
            yield reply_prefix
        current_len = 0
        async for output in self.model.generate(
            prompt, sampling_params=sampling_params, request_id=request_id
        ):
            await asyncio.sleep(0.001)
            yield output.outputs[0].text[current_len:]
            current_len = len(output.outputs[0].text)
            if output.finished:
                break

    async def abort(self, request_id: str):
        """Abort generation"""
        await self.model.abort(request_id)