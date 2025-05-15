from typing import Any

import modal

DEFAULT_MODEL = "gemini/gemini-2.5-pro-preview-03-25"
DEFAULT_CONCURRENCY = 20
DEFAULT_TIMEOUT = 30 * 60 * 4


def get_vlm_modal_image(**env_kwargs: Any) -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.10")
        .pip_install(
            "uvicorn",
            "fastapi",
            "pydantic>=2.0.0",
            "numpy",
            "pillow",
            "litellm",
            "aiohttp",
            "pyyaml",
            "draccus",
            "asyncio",
            "json-numpy",
        )
        .env({k: v for k, v in env_kwargs.items() if v is not None})
        # Install the local package
        .add_local_python_source("mallet")
        .add_local_python_source("modal_servers")
    )


def get_vlm_modal_secrets() -> list[modal.Secret]:
    return [
        modal.Secret.from_name("openai-api-key"),
        modal.Secret.from_name("anthropic-api-key"),
        modal.Secret.from_name("gemini-api-key"),
        modal.Secret.from_name("xai-api-key"),
    ]
