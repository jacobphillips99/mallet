import modal
from fastapi import FastAPI

from modal_servers.vla import HF_CACHE_PATH, HF_CACHE_VOL, get_openvla_server, image

APP_NAME = "openvla-7b-server"
MODEL_PATH = "openvla/openvla-7b"
TIMEOUT = 30 * 60  # catchall-timeout of 30min
CONCURRENCY = 1  # just for testing
GPU = "A10G"

app = modal.App(
    name=APP_NAME,
    image=image,
)


@app.function(
    gpu=GPU,
    timeout=TIMEOUT,
    volumes={HF_CACHE_PATH: HF_CACHE_VOL},
    max_containers=CONCURRENCY,
    scaledown_window=60,  # 1 minute scaledown window; more important for GPU servers since we want to avoid short drops since the spin-up cost is high
)
@modal.asgi_app()
def serve() -> FastAPI:
    server = get_openvla_server(openvla_path=MODEL_PATH)
    return server._create_app()
