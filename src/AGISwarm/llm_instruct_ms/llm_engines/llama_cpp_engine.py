"""LLaMA C++ Engine"""

from typing import Dict, List, cast

from llama_cpp import CreateCompletionStreamResponse, Llama
from PIL import Image
from pydantic import Field
from transformers import PreTrainedTokenizer

from .engine import Engine, SamplingParams


class LlamaCppSamplingParams(SamplingParams):
    """LlamaCpp sampling settings"""

    repetition_penalty: float = Field(default=1.2, description="Repetition penalty")
    frequency_penalty: float = Field(default=0.0, description="Frequency penalty")
    presence_penalty: float = Field(default=0.0, description="Presence penalty")


class LlamaCppEngine(Engine[LlamaCppSamplingParams]):
    """LLM Instruct Model Inference"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        hf_model_name: str,
        tokenizer_name: str | None,
        filename: str,
        n_gpu_layers: int = -1,
        n_ctx: int = 8192,
    ):
        self.llama = Llama.from_pretrained(
            hf_model_name, filename=filename, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx
        )
        self.tokenizer: PreTrainedTokenizer = PreTrainedTokenizer.from_pretrained(
            tokenizer_name or hf_model_name
        )
        self.conversations: Dict[str, List[Dict]] = {}

    def get_sampling_params(self, sampling_params: LlamaCppSamplingParams):
        """Get sampling params"""
        sampling_params_dict = sampling_params.model_dump()
        sampling_params_dict["max_tokens"] = sampling_params_dict.pop("max_new_tokens")
        sampling_params_dict["repeat_penalty"] = sampling_params_dict.pop(
            "repetition_penalty"
        )
        return sampling_params_dict

    async def generate(
        self,
        messages: list[dict],
        image: Image.Image | None = None,
        reply_prefix: str = "",
        sampling_params: LlamaCppSamplingParams = LlamaCppSamplingParams(),
    ):
        """Generate text from prompt"""
        if image:
            raise NotImplementedError("Image input not supported")
        prompt = self.prepare_prompt(self.tokenizer, messages)
        sampling_params_dict = self.get_sampling_params(sampling_params)
        if reply_prefix:
            yield reply_prefix
        for output in self.llama(
            prompt,
            **sampling_params_dict,
            stream=True,
        ):
            output = cast(CreateCompletionStreamResponse, output)
            yield output["choices"][0]["text"]
