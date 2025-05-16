"""
Deploy the VLA Policy Server Modal app with a tunnel.

Usage:

```bash
MODEL="openvla" modal run modal_servers/vla/vla_modal_server_with_tunnel.py
```

This will print our the host and tunnel; should look like `xxx.modal.host:yyyyy`.
Note we use `modal run` in order to manually spin up and tear down the server.

Also, note that the tunnel server is single-container, so "sequential" evaluation must be used.
"""

import os

import modal

from modal_servers.vla import (
    DEFAULT_GPU,
    DEFAULT_TIMEOUT,
    GET_VLA_FUNCTIONS,
    HF_CACHE_PATH,
    HF_CACHE_VOL,
    VLA_MODEL_PATHS,
    get_vla_modal_image,
)

MODEL = os.environ.get("MODEL", "openvla")
APP_NAME = VLA_MODEL_PATHS[MODEL].split("/")[-1].replace("-", "_").replace(".", "_") + "-tunnel"
app = modal.App(
    name=APP_NAME,
    image=get_vla_modal_image(MODEL=MODEL),
)


@app.function(
    gpu=DEFAULT_GPU,
    timeout=DEFAULT_TIMEOUT,
    volumes={HF_CACHE_PATH: HF_CACHE_VOL},
    max_containers=1,  # tunnel server is single-container
    scaledown_window=60,  # 1 minute scaledown window; more important for GPU servers since we want to avoid short drops since the spin-up cost is high
)
def serve_vla_tunnel() -> None:
    server = GET_VLA_FUNCTIONS[MODEL](VLA_MODEL_PATHS[MODEL])
    internal_port = 8000
    with modal.forward(internal_port, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        # Serve the server on a tunnel to expose port; find tcp socket in logs
        print(f"hit me at {host}:{port}")
        server.run(host="0.0.0.0", port=internal_port)
