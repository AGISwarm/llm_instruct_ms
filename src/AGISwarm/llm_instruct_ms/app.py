"""Main module for the LLM instruct microservice"""

import asyncio
import base64
import logging
import re
import traceback
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
            max_concurrent_tasks=5,
            sleep_time=0,
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
        else:
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
                image: Image.Image | None = (
                    self.base64_to_image(gen_config.image) if gen_config.image else None
                )
                # Enqueue the task (without starting it)
                queued_task = self.queue_manager.queued_generator(
                    self.llm_pipeline.__call__,
                    pass_task_id=isinstance(
                        self.llm_pipeline, ConcurrentEngine  # type: ignore
                    ),
                )
                # task_id and interrupt_event are created by the queued_generator
                task_id = queued_task.task_id
                await websocket.send_json(
                    {
                        "status": TaskStatus.STARTING,
                        "task_id": task_id,
                    }
                )
                # Start the generation task
                sampling_dict = self.sampling_settings_cls.model_validate(
                    gen_config,
                    strict=False,
                )
                try:
                    async for step_info in queued_task(
                        conversation_id,
                        gen_config.prompt,
                        gen_config.system_prompt,
                        gen_config.reply_prefix,
                        image,
                        sampling_dict,
                    ):
                        await asyncio.sleep(0)
                        if (
                            not isinstance(step_info, dict) or "status" not in step_info
                        ):  # Task's return value.
                            await websocket.send_json(
                                {
                                    "task_id": task_id,
                                    "status": TaskStatus.RUNNING,
                                    "tokens": step_info,
                                }
                            )
                            continue
                        if (
                            step_info["status"] == TaskStatus.WAITING
                        ):  # Queuing info returned
                            await websocket.send_json(step_info)
                            continue
                        if (
                            step_info["status"] != TaskStatus.RUNNING
                        ):  # Queuing info returned
                            await websocket.send_json(step_info)
                            break
                    await websocket.send_json(
                        {
                            "task_id": task_id,
                            "status": TaskStatus.FINISHED,
                        }
                    )
                except asyncio.CancelledError as e:
                    logging.info(e)
                    await websocket.send_json(
                        {
                            "status": TaskStatus.ABORTED,
                            "task_id": task_id,
                        }
                    )
                except Exception as e:  # pylint: disable=broad-except
                    logging.error(e)
                    traceback.print_exc()
                    await websocket.send_json(
                        {
                            "status": TaskStatus.ERROR,
                            "message": str(e),  ### loggging
                        }
                    )
        except WebSocketDisconnect:
            print("Client disconnected", flush=True)
        finally:
            self.llm_pipeline.conversations.pop(conversation_id, None)
            await websocket.close()

    class AbortRequest(BaseModel):
        """Abort request"""

        task_id: str

    async def abort(self, request: AbortRequest):
        """Abort generation"""
        print(f"ENTER ABORT Aborting request {request.task_id}")
        async with self.start_abort_lock:
            print(f"Aborting request {request.task_id}")
            await self.queue_manager.abort_task(request.task_id)
