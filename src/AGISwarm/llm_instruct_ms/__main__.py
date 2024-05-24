"""Main module for the LLM instruct microservice"""

from argparse import ArgumentParser

import uvicorn

from .app import LLMInstructApp
from .settings import LLMInstructSettings


def main():
    """Main function"""
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the configuration file")

    args = parser.parse_args()
    yaml_path = args.config

    settings = LLMInstructSettings.from_yaml(yaml_path)
    llm_instruct_app = LLMInstructApp(settings)
    uvicorn.run(llm_instruct_app.app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
