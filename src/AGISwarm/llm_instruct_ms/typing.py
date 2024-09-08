"""Application settings"""

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Type, Union

from omegaconf import DictConfig
from uvicorn.config import LoopSetupType

from .llm_engines import (
    HFEngine,
    HFSamplingParams,
    LlamaCppEngine,
    LlamaCppSamplingParams,
    VLLMEngine,
    VLLMSamplingParams,
)

ENGINE_MAP: Dict[str, Type[Union[HFEngine, VLLMEngine, LlamaCppEngine]]] = {
    "HFEngine": HFEngine,
    "VLLMEngine": VLLMEngine,
    "LlamaCppEngine": LlamaCppEngine,
}

ENGINE_SAMPLING_PARAMS_MAP: Dict[
    str, Type[Union[HFSamplingParams, VLLMSamplingParams, LlamaCppSamplingParams]]
] = {
    "HFEngine": HFSamplingParams,
    "VLLMEngine": VLLMSamplingParams,
    "LlamaCppEngine": LlamaCppSamplingParams,
}


@dataclass
class ModelConfig(DictConfig):
    """Model settings"""


@dataclass
class VLLMConfig(ModelConfig):
    """VLLM settings"""


@dataclass
class HFConfig(ModelConfig):
    """HF settings"""


@dataclass
class LlamaCppConfig(ModelConfig):
    """LlamaCpp settings"""

    filename: str = "*F16.gguf"
    n_gpu_layers: int = -1
    n_ctx: int = 8192


ENGINE_CONFIG_MAP: Dict[str, Type] = {
    "HFEngine": HFConfig,
    "VLLMEngine": VLLMConfig,
    "LlamaCppEngine": LlamaCppConfig,
}


@dataclass
class SamplingConfig(DictConfig):
    """Default sampling settings"""

    max_new_tokens: int = 1000
    temperature: float = 0.6
    top_p: float = 0.95
    repetition_penalty: float = 1.2
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


@dataclass
class UvicornConfig(DictConfig):
    """
    A class to hold the configuration for the Uvicorn.
    """

    host: str
    port: int
    log_level: str
    loop: LoopSetupType


@dataclass
class GUIConfig(DictConfig):
    """GUI settings"""

    default_sampling_config: SamplingConfig


@dataclass
class LLMInstructConfig(DictConfig):
    """LLM Instruct settings"""

    hf_model_name: str
    tokenizer_name: str | None
    engine: Literal["HFEngine", "VLLMEngine", "LlamaCppEngine"]
    engine_config: Optional[Union[HFConfig, VLLMConfig, LlamaCppConfig]]
    gui_config: GUIConfig
    uvicorn_config: UvicornConfig
    sampling_settings: SamplingConfig
