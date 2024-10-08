""" LLM Instruct Model Inference """

import asyncio
import logging
from typing import Dict, List, Optional

import vllm  # type: ignore
from huggingface_hub import hf_hub_download
from PIL import Image
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
        **kwargs,
    ):
        if filename is not None:
            model = hf_hub_download(hf_model_name, filename)
        else:
            model = hf_model_name
        self.conversations: Dict[str, List[Dict]] = {}
        self.image: Dict[str, Image.Image | None] = {}
        self.model = vllm.AsyncLLMEngine.from_engine_args(
            vllm.AsyncEngineArgs(
                model=model,
                tokenizer=tokenizer_name or hf_model_name,
                trust_remote_code=True,
                **kwargs,
            )
        )
        logging.info("Model loaded")
        mm_cfg = asyncio.run(self.model.get_model_config()).multimodal_config
        if mm_cfg is None:
            self.image_prompt_enabled = False
        elif len(mm_cfg.limit_per_prompt) == 0:
            self.image_prompt_enabled = False
            logging.warning("Model supports multimodal input but no limits are set")
        else:
            self.image_prompt_enabled = (
                mm_cfg.limit_per_prompt["image"] is not None
                and mm_cfg.limit_per_prompt["image"] > 0
            )
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
        )

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    async def generate(
        self,
        messages: List[Dict[str, str]],
        image: Optional[Image.Image],
        reply_prefix: str,
        sampling_params: VLLMSamplingParams,
        task_id: str,
    ):
        """Generate text from prompt"""
        if image and not self.image_prompt_enabled:
            logging.warning("Image input not supported by this model")
        prompt = self.prepare_prompt(self.tokenizer, messages)  # type: ignore
        vllm_sampling_params = self.get_sampling_params(sampling_params)
        current_len = 0
        if reply_prefix:
            yield reply_prefix
        async for output in self.model.generate(
            (
                vllm.TextPrompt(
                    {"prompt": prompt, "multi_modal_data": {"image": image}}
                )
                if image and self.image_prompt_enabled
                else prompt
            ),
            sampling_params=vllm_sampling_params,
            request_id=task_id,
        ):
            yield output.outputs[0].text[current_len:]
            current_len = len(output.outputs[0].text)
            if output.finished:
                break
