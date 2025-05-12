"""
MODEL="gpt-4o" modal deploy modal_server.py
"""

import os

import modal

from mallet.servers.server import VLMPolicyServer
from modal_servers.vlm import (
    DEFAULT_CONCURRENCY,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    get_vlm_modal_image,
    get_vlm_modal_secrets,
)

# Read from environment variables or use defaults
# this gets around Modal not accepting parameters during deployment
# send it to the image as environment variable instead
MODEL = os.environ.get("MODEL", DEFAULT_MODEL)
HISTORY_LENGTH = os.environ.get("HISTORY_LENGTH", None)
HISTORY_CHOICE = os.environ.get("HISTORY_CHOICE", None)
print(f"Found model: {MODEL}, history length: {HISTORY_LENGTH}, history choice: {HISTORY_CHOICE}")
# Define app name based on model to have unique deployments for different models
APP_NAME = (
    f"vlm-robot-policy-{MODEL.replace('/', '-').replace('.', '-')}-tunnel"
    + (f"-history-{HISTORY_LENGTH}" if HISTORY_LENGTH is not None else "")
    + (f"-history-choice-{HISTORY_CHOICE}" if HISTORY_CHOICE is not None else "")
)

app = modal.App(
    name=APP_NAME,
    image=get_vlm_modal_image(
        MODEL=MODEL,
        HISTORY_LENGTH=HISTORY_LENGTH,
        HISTORY_CHOICE=HISTORY_CHOICE,
        LITELLM_LOG="CRITICAL",
    ),
    secrets=get_vlm_modal_secrets(),
)


@app.function(
    max_containers=DEFAULT_CONCURRENCY,  # this breaks the rate limits, but fine for now
    timeout=DEFAULT_TIMEOUT,
)
def serve_vlm_tunnel() -> None:
    """
    Modal ASGI app that serves the VLM policy server.
    This reuses the complete FastAPI app from VLMPolicyServer.
    """
    # Initialize the VLM policy server with model from environment variable
    model = os.environ.get("MODEL", DEFAULT_MODEL)
    history_length = os.environ.get("HISTORY_LENGTH", None)
    history_choice = os.environ.get("HISTORY_CHOICE", None)
    server = VLMPolicyServer(
        model=model,
        history_length=int(history_length) if history_length is not None else None,
        history_choice=history_choice,
    )
    # Serve the server on a tunnel to expose port; find tcp host, port from socket in logs
    internal_port = 8000
    with modal.forward(internal_port, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        print(f"hit me at {host}:{port}")
        server.run(host="0.0.0.0", port=internal_port)
