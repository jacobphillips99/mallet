import modal

from modal_servers.vla import HF_CACHE_PATH, HF_CACHE_VOL, get_openvla_server, image

APP_NAME = "openvla-7b-server-TUNNEL-TEST"
MODEL_PATH = "openvla/openvla-7b"
CONCURRENCY = 1  # just for testing
GPU = "A10G"
TIMEOUT = 30 * 60  # catchall-timeout of 30min

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
def serve() -> None:
    # import here to protect local dev environment from OpenVLA
    server = get_openvla_server(openvla_path=MODEL_PATH)
    with modal.forward(8000, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        print(f"hit me at {host}:{port}")
        server.run(host="0.0.0.0", port=8000)
