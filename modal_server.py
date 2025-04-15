"""
Modal deployment for VLM-based robot control server.
This file creates a Modal web endpoint for the VLM AutoEval Robot Benchmark server.

Dependencies:
pip install modal

Usage:
modal deploy modal_server.py  # Deploy to Modal
modal serve modal_server.py   # Run locally for development
"""

import os
from datetime import datetime

import modal

from vlm_autoeval_robot_benchmark.server import VLMPolicyServer

# Define environment variables with defaults
DEFAULT_MODEL = "gemini/gemini-2.5-pro-preview-03-25"
DEFAULT_CONCURRENCY = 1
DEFAULT_TIMEOUT = 300

# Define the Modal image
image = (
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
    )
    # Install the local package
    .add_local_python_source("vlm_autoeval_robot_benchmark")
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

app = modal.App(
    name="vlm-robot-policy-server",
    image=image,
    secrets=[
        modal.Secret.from_name("openai-api-key"),
        modal.Secret.from_name("anthropic-api-key"),
        modal.Secret.from_name("gemini-api-key"),
    ],
)


@app.function(
    max_containers=DEFAULT_CONCURRENCY,
    timeout=DEFAULT_TIMEOUT,
)
@modal.asgi_app()
def fastapi_app():
    """
    Modal ASGI app that serves the VLM policy server.
    This reuses the complete FastAPI app from VLMPolicyServer.
    """
    # Initialize the VLM policy server
    server = VLMPolicyServer(model=DEFAULT_MODEL)
    # Create and return the FastAPI app with all routes
    return server._create_app()


# @modal.web_server(port=8000)
# def my_web_server():
#     import subprocess

#     subprocess.Popen("python vlm_autoeval_robot_benchmark/server.py", shell=True)
