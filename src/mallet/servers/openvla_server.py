"""
Adapted from https://github.com/zhouzypaul/auto_eval/blob/main/auto_eval/policy_server/openvla_server.py,
which is adapted from https://github.com/openvla/openvla/blob/main/vla-scripts/deploy.py

This runs on the remote Modal server which has OpenVLA installed via the image.
"""

import json
import logging
import os.path
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

# ruff: noqa: E402
import json_numpy
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

# for local checkpoint loading
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

json_numpy.patch()
# Register OpenVLA model to HF AutoClasses (only needed when using a local vla path)
AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

# === Utilities ===
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_openvla_prompt(instruction: str, openvla_path: Union[str, Path]) -> str:
    if "v01" in str(openvla_path):
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    else:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


# === Server Interface ===
class OpenVLAServer:
    def __init__(
        self,
        openvla_path: Union[str, Path],
        attn_implementation: Optional[str] = "flash_attention_2",
    ) -> None:
        """
        A simple server for OpenVLA models; exposes `/act` to predict an action for a given image + instruction.
            => Takes in {"image": np.ndarray, "instruction": str, "unnorm_key": Optional[str]}
            => Returns  {"action": np.ndarray}
        """
        self.openvla_path, self.attn_implementation = openvla_path, attn_implementation
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        # Load VLA Model using HF AutoClasses
        self.processor = AutoProcessor.from_pretrained(self.openvla_path, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            self.openvla_path,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)

        # [Hacky] Load Dataset Statistics from Disk (if passing a path to a fine-tuned model)
        if os.path.isdir(self.openvla_path):
            with open(Path(self.openvla_path) / "dataset_statistics.json", "r") as f:
                self.vla.norm_stats = json.load(f)

    def predict_action(self, payload: dict[str, Any]) -> JSONResponse:
        try:
            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            # Parse payload components
            if "image" not in payload or "instruction" not in payload:
                raise HTTPException(
                    status_code=400,
                    detail="Missing required fields: image and instruction",
                )

            image, instruction = payload["image"], payload["instruction"]
            unnorm_key = payload.get("unnorm_key", "bridge_orig")
            image = np.array(image).astype(np.uint8)

            # Run VLA Inference
            prompt = get_openvla_prompt(instruction, self.openvla_path)
            try:
                inputs = self.processor(prompt, Image.fromarray(image).convert("RGB")).to(
                    self.device, dtype=torch.bfloat16
                )
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Error processing image: {str(e)}; {traceback.format_exc()}",
                )

            action = self.vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
            action = list(action)
            if double_encode:
                return JSONResponse(json_numpy.dumps(action))
            else:
                return JSONResponse(action)
        except HTTPException:
            raise
        except Exception as e:
            logging.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Server error: {str(e)}\n"
                    "Make sure your request complies with the expected format:\n"
                    "{'image': np.ndarray, 'instruction': str}\n"
                    "You can optionally add `unnorm_key: str` to specify the dataset statistics."
                ),
            )

    def _create_app(self) -> FastAPI:
        app = FastAPI(
            title="OpenVLA Policy Server",
            description="OpenVLA policy server for AutoEval benchmarking",
        )

        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Allows all origins
            allow_credentials=True,
            allow_methods=["*"],  # Allows all methods
            allow_headers=["*"],  # Allows all headers
        )

        # Add health check endpoint
        @app.get("/health")
        async def health_check() -> dict[str, Any]:
            return {
                "status": "healthy",
                "model": self.openvla_path,
                "timestamp": datetime.now().isoformat(),
            }

        # Add a reset endpoint
        @app.post("/reset")
        async def reset() -> dict[str, Any]:
            return {"status": "reset successful"}

        app.post("/act")(self.predict_action)
        return app

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = self._create_app()

        # Configure server with increased timeout and request size limits
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            timeout_keep_alive=60,
            limit_concurrency=None,
        )
        server = uvicorn.Server(config)
        server.run()
