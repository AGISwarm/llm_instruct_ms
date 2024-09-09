""" LLM Instruct Model Inference """

import asyncio
import logging
from typing import Dict, List, cast

import vllm  # type: ignore
from huggingface_hub import hf_hub_download
from pydantic import Field

from .utils import ConcurrentEngineProtocol, SamplingParams, prepare_prompt


class VLLMSamplingParams(SamplingParams):
    """VLLM sampling settings"""

    repetition_penalty: float = Field(default=1.2, description="Repetition penalty")
    frequency_penalty: float = Field(default=0.0, description="Frequency penalty")
    presence_penalty: float = Field(default=0.0, description="Presence penalty")


class VLLMEngine(ConcurrentEngineProtocol[VLLMSamplingParams]):
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
        self.conversations: Dict[str, List[Dict]] = {}

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
        task_id: str,
        messages: list[dict],
        reply_prefix: str | None,
        sampling_params: VLLMSamplingParams,
    ):
        """Generate text from prompt"""
        prompt = prepare_prompt(self.tokenizer, messages, reply_prefix)
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

    # pylint: disable=too-many-arguments
    async def __call__(
        self,
        conversation_id: str,
        prompt: str,
        system_prompt: str,
        reply_prefix: str,
        sampling_params: VLLMSamplingParams,
        task_id: str,
    ):
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        if system_prompt != "":
            self.conversations[conversation_id].append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )
        self.conversations[conversation_id].append({"role": "user", "content": prompt})
        reply: str = ""
        async for response in self.generate(
            task_id,
            self.conversations[conversation_id],
            reply_prefix,
            sampling_params,
        ):
            reply += response
            yield response
        self.conversations[conversation_id].append(
            {"role": "assistant", "content": reply}
        )
        yield ""
