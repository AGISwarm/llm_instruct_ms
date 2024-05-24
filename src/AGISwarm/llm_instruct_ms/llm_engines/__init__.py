"""
This module contains the different LLM engines that can be used to train the LLM model.
"""

from typing import Protocol, runtime_checkable

from .hf_engine import HFEngine, HFSamplingParams
from .vllm_engine import VLLMEngine, VLLMSamplingParams
from .llama_cpp_engine import LlamaCppEngine, LlamaCppSamplingParams

from .utils import EngineProtocol
