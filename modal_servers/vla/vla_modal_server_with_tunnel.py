import modal

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

APP_NAME = DEFAULT_APP_NAME + "-tunnel"

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
def serve_vla_tunnel() -> None:
    # import here to protect local dev environment from OpenVLA
    server = get_openvla_server(openvla_path=DEFAULT_OPENVLA_PATH)
    with modal.forward(8000, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        # Serve the server on a tunnel to expose port; find tcp socket in logs
        print(f"hit me at {host}:{port}")
        server.run(host="0.0.0.0", port=8000)
