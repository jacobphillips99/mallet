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
from pathlib import Path
from typing import Any, Dict

import modal
from fastapi.responses import JSONResponse

from vlm_autoeval_robot_benchmark.server import VLMPolicyServer

# Define environment variables with defaults
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_CONCURRENCY = 2
DEFAULT_TIMEOUT = 300

# Environment variables
MODEL = os.environ.get("VLM_MODEL", DEFAULT_MODEL)
CONCURRENCY = int(os.environ.get("CONCURRENCY_LIMIT", DEFAULT_CONCURRENCY))
TIMEOUT = int(os.environ.get("TIMEOUT", DEFAULT_TIMEOUT))

# Check if rate_limits.yaml exists
rate_limits_path = Path("rate_limits.yaml")
has_rate_limits = rate_limits_path.exists()

# Define the Modal image
image = (
    modal.Image.debian_slim()
    .pip_install(
        "uvicorn",
        "fastapi",
        "pydantic",
        "numpy",
        "pillow",
        "litellm",
        "aiohttp",
        "pyyaml",
        "draccus",
    )
    # Install the local package
    .pip_install(".")
    # Copy necessary files
    .copy_local_file(
        "vlm_autoeval_robot_benchmark/utils/ecot_primitives/action_bounds.json",
        "/root/vlm_autoeval_robot_benchmark/utils/ecot_primitives/action_bounds.json",
    )
    .copy_local_file(
        "vlm_autoeval_robot_benchmark/config/rate_limits.yaml",
        "/root/vlm_autoeval_robot_benchmark/config/rate_limits.yaml",
    )
)


# Create a Modal stub
stub = modal.Stub(
    name="vlm-robot-policy-server",
    image=image,
    secrets=[
        modal.Secret.from_name("openai-api-key"),
        modal.Secret.from_name("anthropic-api-key"),
        modal.Secret.from_name("gemini-api-key"),
    ],
)


# Define the web endpoint
@stub.function(
    concurrency_limit=CONCURRENCY,
    timeout=TIMEOUT,
)
@modal.web_endpoint(method="POST")
async def act(payload: Dict[str, Any]) -> JSONResponse:
    """
    Modal web endpoint for the /act route.
    Takes in an image, instruction, and optional proprioception data.
    Returns a robot action.
    """
    # Initialize the VLM policy server
    server = VLMPolicyServer(model=MODEL)

    # Use the existing predict_action method
    return await server.predict_action(payload)


@stub.function()
@modal.web_endpoint(method="GET")
async def health() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": MODEL,
        "rate_limits_configured": has_rate_limits,
        "concurrency_limit": CONCURRENCY,
        "timeout": TIMEOUT,
    }


@stub.function()
@modal.web_endpoint(method="GET")
async def root() -> Dict[str, Any]:
    """Root endpoint for basic info"""
    return {
        "name": "VLM AutoEval Robot Policy",
        "model": MODEL,
        "status": "running",
        "rate_limits_configured": has_rate_limits,
        "endpoints": {"act": "/act (POST)", "health": "/health (GET)", "root": "/ (GET)"},
    }


# For local development and testing
if __name__ == "__main__":
    # This will be run when using `modal serve modal_server.py`
    print("Starting VLM Robot Policy Server in development mode")
    print(f"Model: {MODEL}")
    print(f"Rate limits configured: {has_rate_limits}")
    print("Visit http://localhost:8000 to test the endpoints")
