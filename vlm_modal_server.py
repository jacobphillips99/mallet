"""
Modal deployment for VLM-based robot control server.
This file creates a Modal web endpoint for the VLM AutoEval Robot Benchmark server.

Dependencies:
pip install modal

Usage:
modal deploy modal_server.py  # Deploy to Modal
modal serve modal_server.py   # Run locally for development

Environment Variables:
MODEL: The VLM model to use (default: "gemini/gemini-2.5-pro-preview-03-25")
CONCURRENCY: Number of containers to run (default: 1)
TIMEOUT: Request timeout in seconds (default: 300)

Example:
MODEL="gpt-4o" modal deploy modal_server.py
"""

import os

import modal
from fastapi import FastAPI

from vlm_autoeval_robot_benchmark.servers.server import VLMPolicyServer

# Define environment variables with defaults
DEFAULT_MODEL = "gemini/gemini-2.5-pro-preview-03-25"
CONCURRENCY = 20
TIMEOUT = 300

# Read from environment variables or use defaults; this gets around Modal not accepting parameters during deployment
# send it to the image instead
MODEL = os.environ.get("MODEL", DEFAULT_MODEL)
print(f"Found model: {MODEL}")

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
        "json-numpy",
    )
    .env({"MODEL": MODEL})
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

# Define app name based on model to have unique deployments for different models
APP_NAME = f"vlm-robot-policy-{MODEL.replace('/', '-').replace('.', '-')}"

app = modal.App(
    name=APP_NAME,
    image=image,
    secrets=[
        modal.Secret.from_name("openai-api-key"),
        modal.Secret.from_name("anthropic-api-key"),
        modal.Secret.from_name("gemini-api-key"),
    ],
)


@app.function(
    max_containers=CONCURRENCY,  # this breaks the rate limits, but fine for now
    timeout=TIMEOUT,
)
@modal.asgi_app()
def fastapi_app() -> FastAPI:
    """
    Modal ASGI app that serves the VLM policy server.
    This reuses the complete FastAPI app from VLMPolicyServer.
    """
    # Initialize the VLM policy server with model from environment variable
    server = VLMPolicyServer(model=os.environ.get("MODEL", DEFAULT_MODEL))
    # Create and return the FastAPI app with all routes
    return server._create_app()
