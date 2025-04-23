"""
MODEL="gpt-4o" modal deploy modal_server.py
"""

import os

import modal
from fastapi import FastAPI

from modal_servers.vlm import (
    DEFAULT_CONCURRENCY,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    get_vlm_modal_image,
    get_vlm_modal_secrets,
)
from vlm_autoeval_robot_benchmark.servers.server import VLMPolicyServer

# Read from environment variables or use defaults
# this gets around Modal not accepting parameters during deployment
# send it to the image instead
MODEL = os.environ.get("MODEL", DEFAULT_MODEL)
print(f"Found model: {MODEL}")

# Define app name based on model to have unique deployments for different models
APP_NAME = f"vlm-robot-policy-{MODEL.replace('/', '-').replace('.', '-')}"

app = modal.App(
    name=APP_NAME,
    image=get_vlm_modal_image(model=MODEL),
    secrets=get_vlm_modal_secrets(),
)


@app.function(
    max_containers=DEFAULT_CONCURRENCY,  # this breaks the rate limits, but fine for now
    timeout=DEFAULT_TIMEOUT,
)
@modal.asgi_app()
def fastapi_app() -> FastAPI:
    """
    Modal ASGI app that serves the VLM policy server.
    This reuses the complete FastAPI app from VLMPolicyServer.
    `model` is pulled from the environment variable, both during deployment and runtime.
    """
    # Initialize the VLM policy server with model from environment variable
    server = VLMPolicyServer(model=os.environ.get("MODEL", DEFAULT_MODEL))
    # Create and return the FastAPI app with all routes
    return server._create_app()
