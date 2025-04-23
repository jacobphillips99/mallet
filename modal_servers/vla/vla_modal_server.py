import modal
from fastapi import FastAPI

from modal_servers.vla import (
    DEFAULT_APP_NAME,
    DEFAULT_CONCURRENCY,
    DEFAULT_GPU,
    DEFAULT_OPENVLA_PATH,
    DEFAULT_TIMEOUT,
    HF_CACHE_PATH,
    HF_CACHE_VOL,
    get_openvla_server,
    image,
)

app = modal.App(
    name=DEFAULT_APP_NAME,
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
    server = get_openvla_server(openvla_path=DEFAULT_OPENVLA_PATH)
    return server._create_app()
