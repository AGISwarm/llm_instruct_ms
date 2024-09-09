"""Main module for the LLM instruct microservice"""

import os
from pathlib import Path

import hydra
import uvicorn

from .app import LLMInstructApp
from .typing import LLMInstructConfig


@hydra.main(
    config_path=str(Path(os.getcwd()) / "config"),
)
def main(config: LLMInstructConfig):
    """Main function"""

    llm_instruct_app = LLMInstructApp(config)
    uvicorn.run(
        llm_instruct_app.app,
        host=config.uvicorn_config.host,
        port=config.uvicorn_config.port,
        log_level=config.uvicorn_config.log_level,
        loop=config.uvicorn_config.loop,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
