"""Application settings"""

from typing import Dict, Literal, Type, Union
from pathlib import Path

from pydantic_settings import BaseSettings
from ruamel.yaml import YAML

from .llm_engines import (
    HFEngine,
    LlamaCppEngine,
    VLLMEngine,
    HFSamplingParams,
    LlamaCppSamplingParams,
    VLLMSamplingParams,
)

ENGINE_MAP: Dict[str, Type] = {
    "HFEngine": HFEngine,
    "VLLMEngine": VLLMEngine,
    "LlamaCppEngine": LlamaCppEngine,
}

ENGINE_SAMPLING_PARAMS_MAP: Dict[str, Type] = {
    "HFEngine": HFSamplingParams,
    "VLLMEngine": VLLMSamplingParams,
    "LlamaCppEngine": LlamaCppSamplingParams,
}


class EngineType(BaseSettings):
    """Engine type"""

    engine: Literal["HFEngine", "VLLMEngine", "LlamaCppEngine"] = "HFEngine"


class ModelSettings(BaseSettings):
    """Model settings"""

    model_name: str = "DevsDoCode/LLama-3-8b-Uncensored"
    tokenizer_name: str | None = None


class VLLMSettings(ModelSettings):
    """VLLM settings"""


class HFSettings(ModelSettings):
    """HF settings"""


class LlamaCppSettings(ModelSettings):
    """LlamaCpp settings"""

    filename: str = "*F16.gguf"
    n_gpu_layers: int = -1
    n_ctx: int = 8192


ENGINE_SETTINGS_MAP: Dict[str, Type] = {
    "HFEngine": HFSettings,
    "VLLMEngine": VLLMSettings,
    "LlamaCppEngine": LlamaCppSettings,
}


class GUISettings(BaseSettings):
    """GUI settings"""

    gui_title: str = "LLM Instruct"


class DefaultSamplingSettings(BaseSettings):
    """Default sampling settings"""

    default_max_new_tokens: int = 1000
    default_temperature: float = 0.6
    default_top_p: float = 0.95
    default_repetition_penalty: float = 1.2
    default_frequency_penalty: float = 0.0
    default_presence_penalty: float = 0.0


class NetworkingSettings(BaseSettings):
    """Application settings"""

    host: str = "127.0.0.1"
    port: int = 8000
    websocket_url: str | None = "wss://8fb7-188-88-138-217.ngrok-free.app/ws"
    abort_url: str | None = "https://8fb7-188-88-138-217.ngrok-free.app/abort"


class LLMInstructSettings(
    ModelSettings, GUISettings, DefaultSamplingSettings, NetworkingSettings, EngineType
):
    """LLM Instruct settings"""

    engine_settings: Union[VLLMSettings, HFSettings, LlamaCppSettings]

    @classmethod
    def from_yaml(cls, path: Path) -> "LLMInstructSettings":
        """Create settings from YAML"""
        with open(path, "r", encoding="utf-8") as file:
            yaml: dict = YAML(typ="safe", pure=True).load(file)
        engine_type = EngineType(engine=yaml["engine"])
        engine_settings = cls.__choose_engine(engine_type).model_dump()
        for key in engine_settings.keys():
            engine_settings[key] = yaml.pop(key, engine_settings[key])
        yaml["engine_settings"] = engine_settings
        return cls(**yaml)

    @classmethod
    def __choose_engine(  # pylint: disable=missing-function-docstring
        cls, engine_type: EngineType
    ):
        engine = engine_type.engine
        return ENGINE_SETTINGS_MAP[engine]()
