""" LLM Instruct Model Inference """

import asyncio
import logging
from typing import Dict, List, cast

import vllm  # type: ignore
from huggingface_hub import hf_hub_download
from pydantic import Field

from .engine import ConcurrentEngine, SamplingParams


class VLLMSamplingParams(SamplingParams):
    """VLLM sampling settings"""

    repetition_penalty: float = Field(default=1.2, description="Repetition penalty")
    frequency_penalty: float = Field(default=0.0, description="Frequency penalty")
    presence_penalty: float = Field(default=0.0, description="Presence penalty")


class VLLMEngine(ConcurrentEngine[VLLMSamplingParams]):
    """LLM Instruct Model Inference using VLLM"""

    def __init__(
        self,
        hf_model_name: str,
        filename: str | None = None,
        tokenizer_name: str | None = None,
    ):
        if filename is not None:
            model = hf_hub_download(hf_model_name, filename)
        else:
            model = hf_model_name
        self.conversations: Dict[str, List[Dict]] = {}
        self.model = vllm.AsyncLLMEngine.from_engine_args(
            vllm.AsyncEngineArgs(
                model=model,
                tokenizer=tokenizer_name or hf_model_name,
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

    async def generate(
        self,
        messages: list[dict],
        reply_prefix: str,
        sampling_params: VLLMSamplingParams,
        task_id: str,
    ):
        """Generate text from prompt"""
        prompt = self.prepare_prompt(self.tokenizer, messages, reply_prefix)
        vllm_sampling_params = self.get_sampling_params(sampling_params)
        current_len = 0
        if reply_prefix:
            yield reply_prefix
        async for output in self.model.generate(
            prompt, sampling_params=vllm_sampling_params, request_id=task_id
        ):
            yield output.outputs[0].text[current_len:]
            current_len = len(output.outputs[0].text)
            if output.finished:
                break
