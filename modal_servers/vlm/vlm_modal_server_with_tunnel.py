"""
MODEL="gpt-4o" modal deploy modal_server.py
"""

import os

import modal

from modal_servers.vlm import get_vlm_modal_image, get_vlm_modal_secrets
from vlm_autoeval_robot_benchmark.servers.server import VLMPolicyServer

# Define environment variables with defaults
DEFAULT_MODEL = "gemini/gemini-2.5-pro-preview-03-25"
CONCURRENCY = 20
TIMEOUT = 30 * 60

# Read from environment variables or use defaults; this gets around Modal not accepting parameters during deployment
# send it to the image instead
MODEL = os.environ.get("MODEL", DEFAULT_MODEL)
print(f"Found model: {MODEL}")

# Define app name based on model to have unique deployments for different models
APP_NAME = f"vlm-robot-policy-{MODEL.replace('/', '-').replace('.', '-')}-tunnel"

app = modal.App(
    name=APP_NAME,
    image=get_vlm_modal_image(model=MODEL),
    secrets=get_vlm_modal_secrets(),
)


@app.function(
    max_containers=CONCURRENCY,  # this breaks the rate limits, but fine for now
    timeout=TIMEOUT,
)
def serve_vlm_tunnel() -> None:
    """
    Modal ASGI app that serves the VLM policy server.
    This reuses the complete FastAPI app from VLMPolicyServer.
    """
    # Initialize the VLM policy server with model from environment variable
    server = VLMPolicyServer(model=os.environ.get("MODEL", DEFAULT_MODEL))
    # Serve the server on a tunnel to expose port; find tcp socket in logs
    with modal.forward(8000, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        print(f"hit me at {host}:{port}")
        server.run(host="0.0.0.0", port=8000)
