"""
Deploy the VLA Policy Server Modal app.

Usage:

```bash
MODEL="openvla" modal deploy modal_servers/vla/vla_modal_server.py
```

This will print out the Modal URL; should look like `Created web function fastapi_app => https://your-name-here.modal.run`.
"""

import os

import modal
from fastapi import FastAPI

from modal_servers.vla import (
    DEFAULT_CONCURRENCY,
    DEFAULT_GPU,
    DEFAULT_SCALEDOWN_WINDOW,
    DEFAULT_TIMEOUT,
    GET_VLA_FUNCTIONS,
    HF_CACHE_PATH,
    HF_CACHE_VOL,
    VLA_MODEL_PATHS,
    get_vla_modal_image,
)

MODEL = os.environ.get("MODEL", "openvla")
print(f"Found model: {MODEL}")

APP_NAME = VLA_MODEL_PATHS[MODEL].split("/")[-1].replace("-", "_").replace(".", "_")
app = modal.App(
    name=APP_NAME,
    image=get_vla_modal_image(MODEL=MODEL),
)


@app.function(
    gpu=DEFAULT_GPU,
    timeout=DEFAULT_TIMEOUT,
    volumes={HF_CACHE_PATH: HF_CACHE_VOL},
    max_containers=DEFAULT_CONCURRENCY,
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
)
@modal.asgi_app()
def serve_vla() -> FastAPI:
    server = GET_VLA_FUNCTIONS[MODEL](VLA_MODEL_PATHS[MODEL])
    return server._create_app()
