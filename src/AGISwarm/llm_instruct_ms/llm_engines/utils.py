"""Utility functions for LLM engines"""

import asyncio
import threading
from typing import (
    Generic,
    Protocol,
    TypeVar,
    cast,
    runtime_checkable,
)

from abc import abstractmethod


from pydantic import BaseModel


class SamplingParams(BaseModel):
    """Sampling settings"""

    max_new_tokens: int = 1000
    temperature: float = 0.6
    top_p: float = 0.95


_SamplingParams_contra = TypeVar(
    "_SamplingParams_contra", bound=SamplingParams, contravariant=True
)


@runtime_checkable
class EngineProtocol(Protocol, Generic[_SamplingParams_contra]):
    """Engine protocol"""

    @abstractmethod
    async def generate(
        self,
        request_id: str,
        messages: list[dict],
        reply_prefix: str,
        sampling_params: _SamplingParams_contra,
    ):
        """Generate text from prompt"""
        yield {"request_id": request_id, "response": "success", "msg": reply_prefix}

    @abstractmethod
    async def abort(self, request_id: str) -> None:
        """Abort generation"""


def prepare_prompt(
    tokenizer,
    messages: list[dict],
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


__ABORT_EVENTS = {}
__QUEUE = []


def abort_generation_request(request_id: str):
    """Abort generation request"""
    if request_id in __ABORT_EVENTS:
        __ABORT_EVENTS[request_id].set()


def generation_request_queued_func(func, wait_time=0.2):
    """Decorator for generation requests"""

    def abort_response(request_id: str):
        return {
            "request_id": request_id,
            "response": "abort",
            "msg": "Generation aborted.",
        }

    def waiting_response(request_id: str):
        """Waiting response"""
        return {
            "request_id": request_id,
            "response": "waiting",
            "msg": f"Waiting for {__QUEUE.index(request_id)} requests to finish...\n",
        }

    async def wrapper(*args, **kwargs):
        request_id = args[1]
        __ABORT_EVENTS[request_id] = threading.Event()
        __QUEUE.append(request_id)
        try:
            while __QUEUE[0] != request_id:
                await asyncio.sleep(wait_time)
                if __ABORT_EVENTS[request_id].is_set():
                    yield abort_response(request_id)
                    return
                yield waiting_response(request_id)
            async for response in func(*args, **kwargs):
                if __ABORT_EVENTS[request_id].is_set():
                    yield abort_response(request_id)
                    return
                yield response
        except asyncio.CancelledError as e:
            print(e)
        finally:
            __QUEUE.remove(request_id)
            __ABORT_EVENTS[request_id].clear()
            __ABORT_EVENTS.pop(request_id)

    return wrapper
