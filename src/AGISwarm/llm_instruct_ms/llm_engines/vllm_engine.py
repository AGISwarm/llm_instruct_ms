""" LLM Instruct Model Inference """

import asyncio
import logging
from typing import cast

import vllm  # type: ignore
from pydantic import Field

from .utils import EngineProtocol, SamplingParams, prepare_prompt


class VLLMSamplingParams(SamplingParams):
    """VLLM sampling settings"""

    repetition_penalty: float = Field(default=1.2, description="Repetition penalty")
    frequency_penalty: float = Field(default=0.0, description="Frequency penalty")
    presence_penalty: float = Field(default=0.0, description="Presence penalty")


class VLLMEngine(EngineProtocol[VLLMSamplingParams]):
    """LLM Instruct Model Inference using VLLM"""

    def __init__(self, model_name: str, tokenizer_name: str | None = None):
        self.model = vllm.AsyncLLMEngine.from_engine_args(
            vllm.AsyncEngineArgs(
                model=model_name,
                tokenizer=tokenizer_name or model_name,
                dtype="float16",
                tensor_parallel_size=2,
                gpu_memory_utilization=1.0,
                trust_remote_code=True,
            )
        )
        logging.info("Model loaded")
        self.tokenizer = asyncio.run(self.model.get_tokenizer())

    def get_sampling_params(
        self, sampling_params: VLLMSamplingParams
    ) -> vllm.SamplingParams:
        """Get sampling params"""
        sampling_params_dict = sampling_params.model_dump()
        sampling_params_dict["max_tokens"] = sampling_params_dict.pop("max_new_tokens")
        return vllm.SamplingParams(
            **sampling_params_dict,
            skip_special_tokens=True,
            truncate_prompt_tokens=True,
            stop_token_ids=[
                cast(int, self.tokenizer.eos_token_id),
                cast(int, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")),
            ],
        )

    async def generate(  # type: ignore
        self,
        request_id: str,
        messages: list[dict],
        reply_prefix: str | None = None,
        sampling_params: VLLMSamplingParams = VLLMSamplingParams(),
    ):
        """Generate text from prompt"""
        prompt = prepare_prompt(self.tokenizer, messages, reply_prefix)
        vllm_sampling_params = self.get_sampling_params(sampling_params)
        if reply_prefix:
            yield {"request_id": request_id, "response": "success", "msg": reply_prefix}
        current_len = 0
        async for output in self.model.generate(
            prompt, sampling_params=vllm_sampling_params, request_id=request_id
        ):
            await asyncio.sleep(0.001)
            yield {
                "request_id": request_id,
                "response": "success",
                "msg": output.outputs[0].text[current_len:],
            }
            current_len = len(output.outputs[0].text)
            if output.finished:
                break

    async def abort(self, request_id: str):
        """Abort generation"""
        await self.model.abort(request_id)
