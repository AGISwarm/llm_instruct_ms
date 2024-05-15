import asyncio
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic_settings import BaseSettings

from . import __version__
from .llm import LLM


class Settings(BaseSettings):
    """Application settings"""

    model_name: str = "RichardErkhov/IlyaGusev_-_saiga_llama3_8b-4bits"
    tokenizer_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    host: str = "0.0.0.0"
    port: int = 8000


class AppFactory:
    """Application factory"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.app = FastAPI()
        self.llm = self.create_llm()
        self._configure_routers()

    def create_llm(self) -> LLM:
        """Create LLM instance"""
        return LLM(
            model_name=self.settings.model_name,
            tokenizer_name=self.settings.tokenizer_name,
        )

    def _configure_routers(self):
        self.app.include_router(self._create_http_router())
        self.app.include_router(self._create_ws_router())

    def _create_http_router(self) -> APIRouter:
        router = APIRouter()

        @router.get("/")
        async def get_root():
            """Root endpoint. Serves GUI.html."""
            return FileResponse(Path(__file__).parent / "GUI.html")

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
                    message = await websocket.receive_text()
                    messages.append({"role": "user", "content": message})
                    for text in self.llm.generate_yields(
                        messages,
                        reply_prefix="Хорошо!",
                        max_new_tokens=1000,
                        temperature=0.6,
                        top_p=0.9,
                        penalty_alpha=0.1,
                    ):
                        await websocket.send_text(text)
                        await asyncio.sleep(0.001)
            except WebSocketDisconnect:
                print("Client disconnected")
            finally:
                await websocket.close()

        return router


if __name__ == "__main__":
    settings = Settings()
    app_factory = AppFactory(settings)
    import uvicorn

    uvicorn.run(app_factory.app, host=settings.host, port=settings.port)
