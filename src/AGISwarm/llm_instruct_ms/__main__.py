"""Main module for the LLM instruct microservice"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Literal

import uvicorn
from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader
from pydantic_settings import BaseSettings

from .llm_engines import VLLMEngine, HFEngine


class ModelSettings(BaseSettings):
    """Model settings"""

    model_name: str = "DevsDoCode/LLama-3-8b-Uncensored"
    engine: Literal["hf", "vllm"] = "vllm"


class GUISettings(BaseSettings):
    """GUI settings"""

    title: str = "LLM Instruct"


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

    host: str = "0.0.0.0"
    port: int = 8000
    websocket_url: str | None = "wss://8fb7-188-88-138-217.ngrok-free.app/ws"


class Settings(ModelSettings, DefaultSamplingSettings, NetworkingSettings, GUISettings):
    """Application settings"""


class AppFactory:
    """Application factory"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.app = FastAPI()
        if settings.engine == "hf":
            self.llm = HFEngine(settings.model_name)
        elif settings.engine == "vllm":
            self.llm = VLLMEngine(settings.model_name)
        else:
            raise ValueError(f"Unknown engine: {settings.engine}")
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
        async def get_root():
            """Root endpoint. Serves gui/index.html"""
            env = Environment(loader=FileSystemLoader(Path(__file__).parent / "gui"))
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
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint"""
            await websocket.accept()
            try:
                messages: List[Dict[str, Any]] = []
                while True:
                    sampling_dict = await websocket.receive_json()
                    if sampling_dict["system_prompt"] != "":
                        messages.append({
                            "role": "system",
                            "content": sampling_dict["system_prompt"],
                        })
                    messages.append(
                        {"role": "user", "content": sampling_dict["prompt"]}
                    )
                    reply = ""
                    async for text in self.llm.generate(
                        messages,
                        max_new_tokens=sampling_dict["max_new_tokens"],
                        reply_prefix=sampling_dict["reply_prefix"],
                        temperature=sampling_dict["temperature"],
                        top_p=sampling_dict["top_p"],
                        repetition_penalty=sampling_dict["repetition_penalty"],
                        frequency_penalty=sampling_dict["frequency_penalty"],
                        presence_penalty=sampling_dict["presence_penalty"],
                    ):
                        reply += text
                        await websocket.send_text(text)
                        await asyncio.sleep(0.001)
                    messages.append({"role": "assistant", "content": reply})
                    await websocket.send_text("<end_of_response>")
            except WebSocketDisconnect:
                print("Client disconnected", flush=True)
            finally:
                await websocket.close()

        return router


def main():
    """Main function"""
    settings = Settings()
    app_factory = AppFactory(settings)
    uvicorn.run(app_factory.app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
