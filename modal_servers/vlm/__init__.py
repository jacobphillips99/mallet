from typing import List

import modal

DEFAULT_MODEL = "gemini/gemini-2.5-pro-preview-03-25"
DEFAULT_CONCURRENCY = 20
DEFAULT_TIMEOUT = 30 * 60


def get_vlm_modal_image(model: str = DEFAULT_MODEL) -> modal.Image:
    return (
        modal.Image.debian_slim()
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
        .env({"MODEL": model})
        # Install the local package
        .add_local_python_source("vlm_autoeval_robot_benchmark")
        .add_local_python_source("modal_servers")
        # Copy necessary files
        .add_local_file(
            "vlm_autoeval_robot_benchmark/utils/ecot_primitives/action_bounds.json",
            "/root/vlm_autoeval_robot_benchmark/utils/ecot_primitives/action_bounds.json",
        )
        .add_local_file(
            "vlm_autoeval_robot_benchmark/config/rate_limits.yaml",
            "/root/vlm_autoeval_robot_benchmark/config/rate_limits.yaml",
        )
    )


def get_vlm_modal_secrets() -> List[modal.Secret]:
    return [
        modal.Secret.from_name("openai-api-key"),
        modal.Secret.from_name("anthropic-api-key"),
        modal.Secret.from_name("gemini-api-key"),
        modal.Secret.from_name("xai-api-key"),
    ]
