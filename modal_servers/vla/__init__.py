from typing import Any

import modal

HF_CACHE_VOL = modal.Volume.from_name("hf-cache", create_if_missing=True)
HF_CACHE_PATH = "/cache"

DEFAULT_OPENVLA_PATH = "openvla/openvla-7b"
DEFAULT_CONCURRENCY = 1  # for testing
DEFAULT_GPU = "A10G"
DEFAULT_TIMEOUT = 30 * 60

DEFAULT_ECOT_PATH = "Embodied-CoT/ecot-openvla-7b-bridge"


def get_openvla_server(openvla_path: str = DEFAULT_OPENVLA_PATH) -> Any:
    # import here to protect local dev environment from OpenVLA requirements
    from vlm_autoeval_robot_benchmark.servers.openvla_server import OpenVLAServer

    # would like to use flash_attention_2 but hitting nvcc issues
    server = OpenVLAServer(
        openvla_path=openvla_path,
        attn_implementation="sdpa",
    )
    return server


def get_ecot_server(ecot_path: str = DEFAULT_ECOT_PATH) -> Any:
    from vlm_autoeval_robot_benchmark.servers.ecot_server import ECOTServer

    server = ECOTServer(ecot_path=ecot_path)
    return server


image = (
    modal.Image.debian_slim()
    .apt_install("git", "python3-opencv")
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
        "opencv-python",
        "git+https://github.com/openvla/openvla.git",
    )
    .add_local_python_source("vlm_autoeval_robot_benchmark")
    .add_local_python_source("modal_servers")
)
