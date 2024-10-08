"""Utility functions for LLM engines"""

import uuid
from abc import abstractmethod
from typing import AsyncGenerator, Dict, Generic, List, Optional, TypeVar, cast

from PIL import Image
from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase


class SamplingParams(BaseModel):
    """Sampling settings"""

    max_new_tokens: int = 1000
    temperature: float = 0.6
    top_p: float = 0.95


_SamplingParams_contra = TypeVar(
    "_SamplingParams_contra", bound=SamplingParams, contravariant=True
)


# pylint: disable=too-few-public-methods
class PreparePromptMixin:
    """Prepare prompt mixin"""

    def prepare_prompt(
        self,
        tokenizer: PreTrainedTokenizerBase,
        messages: List[Dict[str, str]],
    ):
        """Prepare prompt for model"""
        eot_uuid = "eot_" + str(uuid.uuid4())
        prompt = (
            cast(
                str,
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                ),
            ).rstrip()
            + eot_uuid
        )
        prompt = prompt.replace(tokenizer.eos_token + eot_uuid, "")
        prompt = prompt.replace(eot_uuid, "")
        return prompt


# pylint: disable=too-few-public-methods
class Engine(Generic[_SamplingParams_contra], PreparePromptMixin):
    """Engine protocol"""

    conversations: Dict[str, List[Dict[str, str]]]
    image: Dict[str, Image.Image | None]
    image_prompt_enabled: bool

    # pylint: disable=too-many-arguments
    async def __call__(
        self,
        conversation_id: str,
        prompt: str,
        system_prompt: str,
        reply_prefix: str,
        image: Optional[Image.Image],
        sampling_params: _SamplingParams_contra,
    ) -> AsyncGenerator[str, None]:
        if image:
            prompt = "<image>\n" + prompt if image else prompt
            self.image[conversation_id] = image
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
            self.conversations[conversation_id],
            self.image[conversation_id],
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
        image: Optional[Image.Image],
        reply_prefix: str,
        sampling_params: _SamplingParams_contra,
    ):
        """Generate text from prompt"""
        yield str()


# pylint: disable=too-few-public-methods
class ConcurrentEngine(Generic[_SamplingParams_contra], PreparePromptMixin):
    """Concurrent engine protocol"""

    conversations: Dict[str, List[Dict[str, str]]]
    image: Dict[str, Image.Image | None]

    # pylint: disable=too-many-arguments
    async def __call__(
        self,
        conversation_id: str,
        prompt: str,
        system_prompt: str,
        reply_prefix: str,
        image: Optional[Image.Image],
        sampling_params: _SamplingParams_contra,
        task_id: str,
    ):
        reply_prefix = (reply_prefix + " ").strip()
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
            self.image[conversation_id] = None
        if image:
            prompt = "<image>\n" + prompt if image else prompt
            self.image[conversation_id] = image
            for message in self.conversations[conversation_id]:
                if message["role"] == "user":
                    message["content"] = message["content"].replace(
                        "<image>", "<seen_image>"
                    )
        else:
            self.image[conversation_id] = None
        if system_prompt != "":
            self.conversations[conversation_id].append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )
        self.conversations[conversation_id].append({"role": "user", "content": prompt})
        self.conversations[conversation_id].append(
            {"role": "assistant", "content": reply_prefix}
        )
        try:
            async for response in self.generate(
                self.conversations[conversation_id],
                self.image[conversation_id],
                reply_prefix,
                sampling_params,
                task_id,
            ):
                self.conversations[conversation_id][-1]["content"] += response
                yield response
        finally:
            yield ""

    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        image: Optional[Image.Image],
        reply_prefix: str,
        sampling_params: _SamplingParams_contra,
        task_id: str,
    ):
        """Generate text from prompt"""
        yield str()
