"""Main module for the LLM instruct microservice"""

import uuid
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader
from pydantic_settings import BaseSettings

from ..llm_engines import EngineProtocol
from ..settings import ENGINE_MAP, ENGINE_SAMPLING_PARAMS_MAP, LLMInstructSettings


class LLMInstructApp:  # pylint: disable=too-few-public-methods
    """Application factory"""

    def __init__(self, settings: LLMInstructSettings):
        self.settings = settings
        self.app = FastAPI()
        self.llm: EngineProtocol[Any] = ENGINE_MAP[settings.engine](
            **settings.engine_settings.model_dump()
        )
        self.sampling_settings_cls = ENGINE_SAMPLING_PARAMS_MAP[settings.engine]
        self._configure_routers()

    def _configure_routers(self):
        self.app.mount(
            "/static",
            StaticFiles(directory=Path(__file__).parent / "gui"),
            name="static",
        )
        self.app.include_router(self._create_http_router())
        self.app.include_router(self._create_ws_router())

    def _create_http_router(self) -> APIRouter:
        router = APIRouter()

        @router.get("/")
        async def get_root():  # type: ignore
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
                        **self.settings.model_dump(),
                    )
                )
            return FileResponse(Path(__file__).parent / "gui" / "current_index.html")

        return router

    def _create_ws_router(self) -> APIRouter:
        router = APIRouter()

        @router.websocket("/ws")
        async def generate(websocket: WebSocket):  # type: ignore
            """WebSocket endpoint"""
            await websocket.accept()
            try:
                messages: List[Dict[str, Any]] = []
                while True:
                    sampling_dict: Dict[str, Any] = await websocket.receive_json()
                    system_prompt = sampling_dict.pop("system_prompt", "")
                    reply_prefix = sampling_dict.pop("reply_prefix", "")
                    prompt = sampling_dict.pop("prompt", "")
                    if system_prompt != "":
                        messages.append(
                            {
                                "role": "system",
                                "content": system_prompt,
                            }
                        )
                    messages.append({"role": "user", "content": prompt})
                    reply = ""
                    request_id = str(uuid.uuid4())
                    async for response in self.llm.generate(
                        request_id,
                        messages,
                        reply_prefix,
                        self.sampling_settings_cls.model_validate(sampling_dict),
                    ):
                        if response["response"] == "waiting":
                            await websocket.send_json(response)
                        elif response["response"] == "success":
                            reply += response["msg"]
                            await websocket.send_json(response)
                        else:
                            raise ValueError(
                                f"Invalid response: {response['response']}"
                            )
                    messages.append({"role": "assistant", "content": reply})
                    await websocket.send_json(
                        {
                            "request_id": request_id,
                            "response": "end",
                            "msg": "",
                        }
                    )
            except WebSocketDisconnect:
                print("Client disconnected", flush=True)
            finally:
                await websocket.close()

        class AbortRequest(BaseSettings):
            """Abort request"""

            request_id: str

        @router.post("/abort")
        async def abort(request: AbortRequest):
            """Abort generation"""
            await self.llm.abort(request.request_id)

        return router
