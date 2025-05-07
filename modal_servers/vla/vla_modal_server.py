import os

import modal
from fastapi import FastAPI

from modal_servers.vla import (
    DEFAULT_CONCURRENCY,
    DEFAULT_GPU,
    DEFAULT_TIMEOUT,
    GET_VLA_FUNCTIONS,
    HF_CACHE_PATH,
    HF_CACHE_VOL,
    VLA_MODEL_PATHS,
    image,
)

MODEL = os.environ.get("VLA_MODEL", "openvla")
APP_NAME = VLA_MODEL_PATHS[MODEL].split("/")[-1].replace("-", "_").replace(".", "_")
app = modal.App(
    name=APP_NAME,
    image=image,
)


@app.function(
    gpu=DEFAULT_GPU,
    timeout=DEFAULT_TIMEOUT,
    volumes={HF_CACHE_PATH: HF_CACHE_VOL},
    max_containers=DEFAULT_CONCURRENCY,
    scaledown_window=60,  # 1 minute scaledown window; more important for GPU servers since we want to avoid short drops since the spin-up cost is high
)
@modal.asgi_app()
def serve_vla() -> FastAPI:
    # lazy import OpenVLAServer to protect local dev environment from OpenVLA requirements
    server = GET_VLA_FUNCTIONS[MODEL](VLA_MODEL_PATHS[MODEL])
    # server = get_ecot_server(ecot_path=DEFAULT_ECOT_PATH)
    return server._create_app()
