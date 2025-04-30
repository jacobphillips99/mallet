# simpler_env_app.py
#
# modal: modal deploy simpler_env_app.py
# example run:
#   modal run simpler_env_app.py::run_script --script rt1_drawer_visual_matching.sh
#
import os
from pathlib import Path
from typing import List, Optional

import modal

REPO_URL: str = "https://github.com/youliangtan/SimplerEnv.git"
REPO_DIR: str = "/root/SimplerEnv"


def _image() -> modal.Image:
    cuda_version = "12.8.0"  # should be no greater than host CUDA version
    flavor = "devel"  #  includes full CUDA toolkit
    operating_sys = "ubuntu22.04"
    tag = f"{cuda_version}-{flavor}-{operating_sys}"

    image = (
        modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
        .apt_install(
            "git",
            "ffmpeg",
            "build-essential",
            "g++",
            "cmake",
            "clang",
            "vulkan-tools",
            "libvulkan-dev",
            "libvulkan1",
        )
        .pip_install("numpy==1.24.4", "jupyterlab")
        .env({"CC": "gcc", "CXX": "g++"})
        .run_commands(
            f"git clone {REPO_URL} --recurse-submodules /root/SimplerEnv",
            "cd /root/SimplerEnv/ManiSkill2_real2sim && pip install -e .",
            "cd /root/SimplerEnv && pip install -r requirements_full_install.txt && pip install -e .",
        )
    )
    return image


app = modal.App("simpler-env")
img = _image()


@app.function(
    image=img,
    gpu="A10G",
    timeout=60 * 60,
)
def main() -> None:
    import subprocess

    print(f"{os.listdir('SimplerEnv')}")

    result = subprocess.run(
        ["python", "/root/SimplerEnv/eval_simpler.py", "--test", "--env", "widowx_open_drawer"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)


# @app.function()
# def run_jupyter() -> None:
#     import secrets
#     import subprocess

#     token = secrets.token_urlsafe(16)
#     with modal.forward(8888) as t:
#         # deep-link straight to the notebook
#         url = f"{t.url}/lab/tree/SimplerEnv/example.ipynb?token={token}"
#         print("open this in your browser rn:", url, flush=True)

#         subprocess.run(
#             [
#                 "jupyter",
#                 "lab",
#                 "--no-browser",
#                 "--ip=0.0.0.0",
#                 "--port=8888",
#                 "--LabApp.allow_origin='*'",
#                 "--LabApp.allow_remote_access=1",
#             ],
#             env={**os.environ, "JUPYTER_TOKEN": token},
#         )
