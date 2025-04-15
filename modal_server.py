"""
Modal deployment for VLM-based robot control server.
This file creates a Modal web endpoint for the VLM AutoEval Robot Benchmark server.

Dependencies:
pip install modal

Usage:
modal deploy modal_server.py  # Deploy to Modal
modal serve modal_server.py   # Run locally for development
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import modal
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from vlm_autoeval_robot_benchmark.server import VLMPolicyServer

# Define environment variables with defaults
DEFAULT_MODEL = "gemini/gemini-2.5-pro-preview-03-25"
DEFAULT_CONCURRENCY = 1
DEFAULT_TIMEOUT = 300
PORT = 8000

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

# Initialize the VLM policy server globally
server = VLMPolicyServer(model=DEFAULT_MODEL)

# Create a FastAPI app
web_app = FastAPI(
    title="VLM AutoEval Robot Policy",
    description="VLM-based robot policy server for AutoEval benchmarking",
)

# Add CORS middleware
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Add the /act route
@web_app.post("/act")
async def act_endpoint(payload: Dict[str, Any]) -> JSONResponse:
    """
    Process an action request.
    Takes in an image, instruction, and optional proprioception data.
    Returns a robot action.
    """
    return await server.predict_action(payload)


# Add health check endpoint
@web_app.get("/health")
async def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "model": DEFAULT_MODEL,
        "timestamp": datetime.now().isoformat(),
    }


# Add root endpoint for basic info
@web_app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "name": "VLM AutoEval Robot Policy",
        "model": DEFAULT_MODEL,
        "status": "running",
        "timestamp": datetime.now().isoformat(),
    }


# Define the web endpoint using the FastAPI app
@app.function(
    max_containers=DEFAULT_CONCURRENCY,
    timeout=DEFAULT_TIMEOUT,
)
@modal.asgi_app()
def fastapi_app():
    """
    Modal ASGI app that serves the VLM policy server with all routes (/act, /health, /).
    This ensures the server is accessible at the expected paths.
    """
    return web_app
