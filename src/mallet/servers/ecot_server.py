"""
Taken from https://github.com/MichalZawalski/embodied-CoT/
https://colab.research.google.com/drive/1CzRKin3T9dl-4HYBVtuULrIskpVNHoAH?usp=sharing#scrollTo=owVajjweDopA
"""

import enum
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Union

import cv2

# ruff: noqa: E402
import json_numpy
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

json_numpy.patch()

# === Utilities ===
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

ECOT_PATH = "Embodied-CoT/ecot-openvla-7b-bridge"


def get_openvla_prompt(instruction: str, openvla_path: Union[str, Path]) -> str:
    if "v01" in str(openvla_path):
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    else:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def split_reasoning(text: str, tags: list[str]) -> dict[str, str]:
    new_parts: dict[Any, Any] = {None: text}

    for tag in tags:
        parts = new_parts
        new_parts = dict()

        for k, v in parts.items():
            if tag in v:
                s = v.split(tag)
                new_parts[k] = s[0]
                new_parts[tag] = s[1]
                # print(tag, s)
            else:
                new_parts[k] = v

    return new_parts


class CotTag(enum.Enum):
    TASK = "TASK:"
    PLAN = "PLAN:"
    VISIBLE_OBJECTS = "VISIBLE OBJECTS:"
    SUBTASK_REASONING = "SUBTASK REASONING:"
    SUBTASK = "SUBTASK:"
    MOVE_REASONING = "MOVE REASONING:"
    MOVE = "MOVE:"
    GRIPPER_POSITION = "GRIPPER POSITION:"
    ACTION = "ACTION:"


def get_cot_tags_list() -> list[str]:
    return [
        CotTag.TASK.value,
        CotTag.PLAN.value,
        CotTag.VISIBLE_OBJECTS.value,
        CotTag.SUBTASK_REASONING.value,
        CotTag.SUBTASK.value,
        CotTag.MOVE_REASONING.value,
        CotTag.MOVE.value,
        CotTag.GRIPPER_POSITION.value,
        CotTag.ACTION.value,
    ]


def resize_pos(pos: list[int], img_size: tuple[int, int]) -> list[int]:
    return [(x * size) // 256 for x, size in zip(pos, img_size)]


def name_to_random_color(name: str) -> list[int]:
    return [(hash(name) // (256**i)) % 256 for i in range(3)]


def draw_gripper(
    img: np.ndarray, pos_list: list[list[int]], img_size: tuple[int, int] = (640, 480)
) -> None:
    for i, pos in enumerate(reversed(pos_list)):
        pos = resize_pos(pos, img_size)
        scale = 255 - int(255 * i / len(pos_list))
        cv2.circle(img, pos, 6, (0, 0, 0), -1)
        cv2.circle(img, pos, 5, (scale, scale, 255), -1)


def get_metadata(reasoning: dict[Any, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {"gripper": [[0, 0]], "bboxes": dict[str, list[int]]()}

    if f" {CotTag.GRIPPER_POSITION.value}" in reasoning:
        gripper_pos = reasoning[f" {CotTag.GRIPPER_POSITION.value}"]
        gripper_pos = gripper_pos.split("[")[-1]
        gripper_pos = gripper_pos.split("]")[0]
        gripper_pos = [int(x) for x in gripper_pos.split(",")]
        gripper_pos = [
            (gripper_pos[2 * i], gripper_pos[2 * i + 1]) for i in range(len(gripper_pos) // 2)
        ]
        metadata["gripper"] = gripper_pos

    if f" {CotTag.VISIBLE_OBJECTS.value}" in reasoning:
        for sample in reasoning[f" {CotTag.VISIBLE_OBJECTS.value}"].split("]"):
            obj = sample.split("[")[0]
            if obj == "":
                continue
            coords = [int(n) for n in sample.split("[")[-1].split(",")]
            metadata["bboxes"][obj] = coords

    return metadata


# === Server Interface ===
class ECOTServer:
    def __init__(
        self,
        ecot_path: Union[str, Path] = ECOT_PATH,
    ) -> None:
        """
        A simple server for ECOT models; exposes `/act` to predict an action for a given image + instruction.
            => Takes in {"image": np.ndarray, "instruction": str, "unnorm_key": Optional[str]}
            => Returns  {"action": np.ndarray}
        """
        self.ecot_path = ecot_path
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        # Load VLA Model using HF AutoClasses
        self.processor = AutoProcessor.from_pretrained(self.ecot_path, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            self.ecot_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)

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

            # Run VLA Inference
            prompt = get_openvla_prompt(instruction, self.ecot_path)
            try:
                inputs = self.processor(prompt, Image.fromarray(image).convert("RGB")).to(
                    self.device, dtype=torch.bfloat16
                )
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

            action, generated_ids = self.vla.predict_action(
                **inputs, unnorm_key=unnorm_key, do_sample=False, max_new_tokens=1024
            )
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
            title="ECOT Policy Server",
            description="ECOT policy server for AutoEval benchmarking",
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
                "model": self.ecot_path,
                "timestamp": datetime.now().isoformat(),
            }

        app.post("/act")(self.predict_action)
        return app

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = self._create_app()

        # Configure server with increased timeout and request size limits
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            timeout_keep_alive=120,
            limit_concurrency=2,
        )
        server = uvicorn.Server(config)
        server.run()
