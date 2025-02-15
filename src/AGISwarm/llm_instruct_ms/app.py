"""Main module for the LLM instruct microservice"""

import asyncio
import base64
import logging
import re
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, cast

from AGISwarm.asyncio_queue_manager import AsyncIOQueueManager, TaskStatus
from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader
from omegaconf import OmegaConf
from PIL import Image
from pydantic import BaseModel

from .llm_engines import ConcurrentEngine, Engine
from .typing import (
    ENGINE_MAP,
    ENGINE_SAMPLING_PARAMS_MAP,
    LLMInstructConfig,
    SamplingConfig,
)


class LLMInstructApp:  # pylint: disable=too-few-public-methods
    """Application factory"""

    def __init__(self, config: LLMInstructConfig):
        self.config = config
        self.app = FastAPI()
        if config.engine_config is None:
            config.engine_config = cast(None, OmegaConf.create())
        self.llm_pipeline: Engine[Any] = ENGINE_MAP[config.engine](  # type: ignore
            hf_model_name=config.hf_model_name,
            tokenizer_name=config.tokenizer_name,
            **cast(dict, OmegaConf.to_container(config.engine_config)),
        )
        self.sampling_settings_cls = ENGINE_SAMPLING_PARAMS_MAP[config.engine]
        self.queue_manager = AsyncIOQueueManager(
            max_concurrent_tasks=2,
            sleep_time=0.001,
        )
        self.start_abort_lock = asyncio.Lock()
        self.setup_routes()

    def setup_routes(self):
        """
        Set up the routes for the Text2Imag e service.
        """
        self.app.get("/", response_class=HTMLResponse)(self.gui)
        self.app.mount(
            "/static",
            StaticFiles(directory=Path(__file__).parent / "gui", html=True),
            name="static",
        )
        self.ws_router = APIRouter()
        self.ws_router.add_websocket_route("/ws", self.generate)
        self.app.post("/abort")(self.abort)
        self.app.include_router(self.ws_router)

    async def gui(self):
        """Root endpoint. Serves gui/index.html"""
        env = Environment(
            loader=FileSystemLoader(Path(__file__).parent / "gui"), autoescape=True
        )
        template = env.get_template("jinja2.html")
        with open(
            Path(__file__).parent / "gui" / "current_index.html",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(
                template.render(
                    OmegaConf.to_container(
                        self.config.gui_config.default_sampling_config
                    ),
                )
            )
        return FileResponse(Path(__file__).parent / "gui" / "current_index.html")

    @staticmethod
    def remove_mime_header(image_data):
        """
        Remove the MIME type header from the image data and return the raw base64 data.
        :return: the raw base64 data
        """
        # Regular expression to match the MIME type header
        mime_pattern = r"^data:image/([a-zA-Z]+);base64,"
        match = re.match(mime_pattern, image_data)

        if match:
            # Remove the header to get the raw base64 data
            base64_data = re.sub(mime_pattern, "", image_data)
            return base64_data
        # If there's no match, assume it's already raw base64 data
        return image_data

    def base64_to_image(self, image: str) -> Image.Image:
        """Convert base64 image to PIL image"""
        image = self.remove_mime_header(image)
        return Image.open(BytesIO(base64.b64decode(image))).convert("RGB")

    async def generate(self, websocket: WebSocket):  # type: ignore
        """WebSocket endpoint"""
        await websocket.accept()
        conversation_id = str(uuid.uuid4())
        try:
            while True:
                data: Dict[str, Any] = await websocket.receive_json()
                gen_config = SamplingConfig(data)
                sampling_dict = self.sampling_settings_cls.model_validate(
                    gen_config,
                    strict=False,
                )
                image: Image.Image | None = (
                    self.base64_to_image(gen_config.image) if gen_config.image else None
                )
                # Enqueue the task (without starting it)
                queued_task = self.queue_manager.queued_task(
                    self.llm_pipeline.__call__,
                    pass_task_id=isinstance(
                        self.llm_pipeline, ConcurrentEngine  # type: ignore
                    ),
                    warnings=(
                        ["Image input not supported by this model"]
                        if image and not self.llm_pipeline.image_prompt_enabled
                        else None
                    ),
                    raise_on_error=False,
                    print_error_tracebacks=True,
                )
                # task_id and interrupt_event are created by the queued_generator
                async for step_info in queued_task(
                    conversation_id,
                    gen_config.prompt,
                    gen_config.system_prompt,
                    gen_config.reply_prefix,
                    image,
                    sampling_dict,
                ):
                    if step_info["status"] == TaskStatus.ERROR:
                        step_info["content"] = None
                    await websocket.send_json(step_info)
        except WebSocketDisconnect:
            logging.info("Client %s disconnected", conversation_id)
        finally:
            self.llm_pipeline.conversations.pop(conversation_id, None)
            await websocket.close()

    class AbortRequest(BaseModel):
        """Abort request"""

        task_id: str

    async def abort(self, request: AbortRequest):
        """Abort generation"""
        async with self.start_abort_lock:
            logging.info("Aborting task %s", request.task_id)
            await self.queue_manager.abort_task(request.task_id)
