import modal

APP_NAME = "openvla-7b-server-TUNNEL-TEST"
MODEL_PATH = "openvla/openvla-7b"
CONCURRENCY = 1  # just for testing
GPU = "A10G"
TIMEOUT = 30 * 60  # catchall-timeout of 30min

HF_CACHE_VOL = modal.Volume.from_name("hf-cache", create_if_missing=True)
HF_CACHE_PATH = "/cache"

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .env(
        {
            "HF_HOME": HF_CACHE_PATH,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
    .pip_install(
        "torch~=2.2",
        "transformers>=4.40",
        "accelerate",
        "bitsandbytes",
        "huggingface_hub",
        "hf_transfer",
        "json-numpy",
        "draccus",
        "pillow",
        "uvicorn",
        "fastapi",
        "pydantic>=2.0.0",
        "numpy",
        "litellm",
        "aiohttp",
        "pyyaml",
        "asyncio",
        "git+https://github.com/openvla/openvla.git",
    )
    .add_local_python_source("vlm_autoeval_robot_benchmark")
)

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
    from vlm_autoeval_robot_benchmark.servers.openvla_server import OpenVLAServer

    server = OpenVLAServer(
        openvla_path=MODEL_PATH,
        attn_implementation="sdpa",  # would like to use flash_attention_2 but hitting nvcc issues
    )
    with modal.forward(8000, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        print(f"hit me at {host}:{port}")
        server.run(host="0.0.0.0", port=8000)
