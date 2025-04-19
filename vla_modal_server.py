import modal
from fastapi import FastAPI

APP_NAME = "openvla-7b-server"
MODEL_PATH = "openvla/openvla-7b"
TIMEOUT = 30 * 60  # sec
CONCURRENCY = 1  # just for testing
GPU = "A10G"

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
)
@modal.asgi_app()
def fastapi_app() -> FastAPI:
    # import here to protect local environment from OpenVLA
    from vlm_autoeval_robot_benchmark.servers.openvla_server import OpenVLAServer

    server = OpenVLAServer(
        openvla_path=MODEL_PATH,
        attn_implementation="sdpa",  # would like to use flash_attention_2 but hitting nvcc issues
    )
    return server._create_app()
