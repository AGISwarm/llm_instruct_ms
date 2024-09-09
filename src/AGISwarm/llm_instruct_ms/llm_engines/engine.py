"""Utility functions for LLM engines"""

from abc import abstractmethod
from typing import Dict, Generic, List, TypeVar, cast

from pydantic import BaseModel


class SamplingParams(BaseModel):
    """Sampling settings"""

    max_new_tokens: int = 1000
    temperature: float = 0.6
    top_p: float = 0.95


_SamplingParams_contra = TypeVar(
    "_SamplingParams_contra", bound=SamplingParams, contravariant=True
)


# pylint: disable=too-few-public-methods
class Engine(Generic[_SamplingParams_contra]):
    """Engine protocol"""

    conversations: Dict[str, List[Dict[str, str]]]

    # pylint: disable=too-many-arguments
    async def __call__(
        self,
        conversation_id: str,
        prompt: str,
        system_prompt: str,
        reply_prefix: str,
        sampling_params: _SamplingParams_contra,
    ):
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

    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        reply_prefix: str,
        sampling_params: _SamplingParams_contra,
    ):
        """Generate text from prompt"""
        yield str()


# pylint: disable=too-few-public-methods
class ConcurrentEngine(Generic[_SamplingParams_contra]):
    """Concurrent engine protocol"""

    conversations: Dict[str, List[Dict[str, str]]]

    # pylint: disable=too-many-arguments
    async def __call__(
        self,
        conversation_id: str,
        prompt: str,
        system_prompt: str,
        reply_prefix: str,
        sampling_params: _SamplingParams_contra,
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
            self.conversations[conversation_id], reply_prefix, sampling_params, task_id
        ):
            reply += response
            yield response
        self.conversations[conversation_id].append(
            {"role": "assistant", "content": reply}
        )
        yield ""

    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        reply_prefix: str,
        sampling_params: _SamplingParams_contra,
        task_id: str,
    ):
        """Generate text from prompt"""
        yield str()


def prepare_prompt(
    tokenizer: object,
    messages: List[Dict[str, str]],
    reply_prefix: str | None = None,
    tokenize: bool = False,
):
    """Prepare prompt for model"""
    if reply_prefix == "":
        reply_prefix = None
    prompt = cast(
        str,
        tokenizer.apply_chat_template(  # type: ignore
            messages,
            tokenize=tokenize,
            add_generation_prompt=reply_prefix is None,
        ),
    ) + ("$" if reply_prefix else "")
    if reply_prefix:
        prompt = prompt.replace("<|eot_id|>$", "<|eot_id|>assistant\n\n" + reply_prefix)

    return prompt
