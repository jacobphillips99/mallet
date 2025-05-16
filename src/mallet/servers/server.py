"""
Lightweight server implementation for VLM-based robot control over REST API.
This server translates VLM outputs to robot actions using the AutoEval format.

Dependencies:
pip install uvicorn fastapi numpy base64 json

Usage:
python vlm_policy_server.py --port 8000

To make your server accessible on the open web, you can use ngrok or bore.pub
With ngrok:
  ngrok http 8000
With bore.pub:
  bore local 8000 --to bore.pub

Note that if you aren't able to resolve bore.pub's DNS (test this with `ping bore.pub`), you can use their actual IP: 159.223.171.199
"""

import asyncio
import base64
import io
import json
import logging
import traceback
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import draccus
import json_numpy
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from mallet.models.rate_limit import rate_limiter
from mallet.models.translation import (
    HISTORY_PREFIX,
    HISTORY_SUFFIX,
    PromptTemplate,
    build_standard_prompt,
)
from mallet.models.vlm import VLM, ImageInput, VLMHistory, VLMInput, VLMRequest, parse_vlm_response
from mallet.utils.ecot_primitives.inverse_ecot_primitive_movements import text_to_move_vector
from mallet.utils.robot_utils import GRIPPER_INDEX, GRIPPER_OPEN_THRESHOLD, get_gripper_position

json_numpy.patch()

logger = logging.getLogger(__name__)


def image_server_helper(image: Any) -> str:
    # Convert image to base64 string
    if isinstance(image, (np.ndarray, list)):
        # Convert list to numpy array if needed
        if isinstance(image, list):
            image = np.array(image, dtype=np.uint8)

        # Convert numpy array to base64
        pil_image = Image.fromarray(image.astype(np.uint8))
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    elif isinstance(image, str):
        # Assume it's already a base64 string
        image_b64 = image
    else:
        raise ValueError(f"Invalid image type: {type(image)}:\n\n{image}")

    return image_b64


# === Server Interface ===
class VLMPolicyServer:
    """
    A server for VLM-based robot policy; exposes `/act` to predict an action for a given image + instruction.
        => Takes in {"image": np.ndarray, "instruction": str, "proprio": Optional[np.ndarray]}
        => Returns  np.ndarray or list[float] of 7-dim action
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        history_length: Optional[int] = None,
        history_choice: Optional[Any] = None,
    ) -> None:
        """
        Initialize the VLM-based policy server

        Args:
            model: The VLM model to use
        """
        # Initialize VLM instance
        self.vlm = VLM()
        self.model = model
        self.history_length = history_length
        self.history_choice = history_choice
        if self.history_length is not None and self.history_choice is not None:
            if isinstance(self.history_choice, int):
                assert (
                    abs(self.history_choice) <= self.history_length
                ), f"history_choice {self.history_choice} must be less than or equal to history_length {self.history_length}"
            elif isinstance(self.history_choice, str):
                assert self.history_choice in [
                    "all"  # todo more options
                ], f"history_choice {self.history_choice} must be in ['all']"
        logger.info(
            f"VLM Policy Server initialized with model: {self.model}{f' and history_length: {self.history_length}' if self.history_length is not None else ''}"
        )
        self.history: Optional[deque[VLMInput]] = None
        if self.history_length is not None:
            self.history = deque(maxlen=self.history_length)

    def setup_history_from_server(self) -> VLMHistory:
        # constructs the VLMHistory object from the server's history
        assert (
            self.history_length is not None and self.history is not None
        ), "history_length must be set if history_flag is True"
        if isinstance(self.history_choice, int):
            history_inputs = [self.history[self.history_choice]]
        elif self.history_choice == "all" or self.history_choice is None:
            history_inputs = list(self.history)
        elif isinstance(self.history_choice, list):
            history_inputs = [self.history[i] for i in self.history_choice]
        else:
            raise ValueError(f"Invalid history choice: {self.history_choice}")
        logger.info(
            f"Using history choice {self.history_choice} to select {len(history_inputs)} history inputs"
        )
        vlm_history = VLMHistory(
            prefix=HISTORY_PREFIX,
            vlm_inputs=history_inputs,
            suffix=HISTORY_SUFFIX,
        )
        return vlm_history

    def setup_history_from_payload(self, payload_history: dict[str, Any]) -> VLMHistory:
        # constructs the VLMHistory object from the payload
        history_inputs = []
        for step in payload_history["steps"]:
            images = [
                ImageInput(data=image_server_helper(image), mime_type="image/png")
                for image in step["images"]
            ]
            history_inputs.append(VLMInput(prompt=step["description"], images=images))
        vlm_history = VLMHistory(
            prefix=HISTORY_PREFIX,
            vlm_inputs=history_inputs,
            suffix=HISTORY_SUFFIX,
        )
        return vlm_history

    def prep_request(
        self,
        image: Any,
        instruction: str,
        raw_proprio: Optional[np.ndarray],
        payload_history: Optional[dict[str, Any]],  # only for testing
    ) -> VLMRequest:
        # Prepare the VLM request
        vlm_image = ImageInput(data=image_server_helper(image), mime_type="image/png")
        logger.info(
            f"proprio (len {len(raw_proprio)}): {raw_proprio}"
            if raw_proprio is not None
            else "proprio is None"
        )
        gripper_position = (
            None if raw_proprio is None else get_gripper_position(raw_proprio[GRIPPER_INDEX])
        )
        logger.info(
            f"gripper_position: {gripper_position} w/ threshold {GRIPPER_OPEN_THRESHOLD} and value {raw_proprio[GRIPPER_INDEX]}"
            if raw_proprio is not None
            else "gripper_position is None"
        )

        history_server_flag = (
            self.history_length is not None and self.history is not None and len(self.history) > 0
        )
        history_payload_flag = (
            payload_history is not None
            and "steps" in payload_history
            and len(payload_history["steps"]) > 0
        )  # for testing

        # Create a prompt for the VLM
        prompt = build_standard_prompt(
            prompt_template=PromptTemplate(
                env_desc="You are looking at a robotics environment.",
                task_desc=instruction,
                gripper_position=gripper_position,
                history_flag=history_server_flag or history_payload_flag,
            )
        )

        # Create history context if available
        vlm_history = None
        if history_server_flag:
            vlm_history = self.setup_history_from_server()
        elif history_payload_flag and payload_history is not None:
            vlm_history = self.setup_history_from_payload(payload_history)

        # Create and send the VLM request
        vlm_request = VLMRequest(
            vlm_input=VLMInput(
                prompt=prompt,
                images=[vlm_image],
            ),
            model=self.model,
            history=vlm_history,
        )
        return vlm_request

    def add_to_history(
        self, vlm_request: VLMRequest, description: str, move_dict: dict[str, Any]
    ) -> None:
        # clean the description
        _ = description.split("**Output:**")[0]
        move_dict_str = json.dumps(move_dict, indent=2)
        caption = f"Movement dictionary from this historical step: {move_dict_str}"
        history_vlm_input = VLMInput(prompt=caption, images=vlm_request.vlm_input.images)
        assert self.history is not None, "history must be set"
        self.history.append(history_vlm_input)

    async def predict_action(self, payload: dict[str, Any]) -> JSONResponse:
        """
        Predict a 7-dim action given an image + proprio + instruction
        """
        try:
            if "encoded" in payload:
                # Support cases where numpy arrays are "double-encoded" as strings
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            # Parse payload components
            if "image" not in payload or "instruction" not in payload:
                raise HTTPException(
                    status_code=400,
                    detail="Missing required fields: image and instruction",
                )
            logger.info(
                f"\n\nNEW REQUEST: history of len {len(self.history) if self.history is not None else 'None'}"
            )
            image = payload["image"]
            instruction = payload["instruction"]
            raw_proprio = payload.get("proprio", None)  # proprio is optional
            payload_history = payload.get("history", None)  # for testing purposes

            vlm_request = self.prep_request(
                image=image,
                instruction=instruction,
                raw_proprio=raw_proprio,
                payload_history=payload_history,
            )

            if instruction == "test connection":
                # short-circuit the VLM call for testing purposes; should help with latency on AutoEval server
                return JSONResponse(content=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            response = await self.vlm.generate(vlm_request)
            try:
                # Extract the JSON part of the response, convert to move vector
                description, move_dict = parse_vlm_response(response.text)
                action = text_to_move_vector(move_dict).tolist()
                logger.info(f"VLM description: {description}")
                logger.info(f"VLM move dict: {move_dict}")
                logger.info(f"VLM action: {action}")

                # optionally add img, description, and move_dict to history
                if self.history_length is not None:
                    self.add_to_history(vlm_request, description, move_dict)

                # Return the action as a JSON response, potentially including the VLM response
                if payload.get("test", False):
                    return JSONResponse(content=dict(action=action, vlm_response=response.text))
                else:
                    return JSONResponse(content=action)

            except Exception as e:
                logger.error(f"Error parsing VLM response: {str(e)}, {traceback.format_exc()}")
                logger.error(f"VLM response: {response.text}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to parse VLM response: {str(e)}",
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=(
                    "Error processing request. "
                    "Make sure your request complies with the expected format:\n"
                    "{'image': np.ndarray/base64, 'instruction': str, 'proprio': Optional[np.ndarray]}\n"
                    f"Error: {str(e)}; Traceback: {traceback.format_exc()}"
                ),
            )

    def _create_app(self) -> FastAPI:
        """
        Create and configure a FastAPI app with all routes.
        This is exposed as a separate method to allow reuse by other deployments (e.g., Modal).

        Returns:
            FastAPI: The configured FastAPI application
        """
        app = FastAPI(
            title="VLM AutoEval Robot Policy",
            description="VLM-based robot policy server for AutoEval benchmarking",
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
                "model": self.model,
                "timestamp": datetime.now().isoformat(),
            }

        @app.get("/")
        async def root() -> dict[str, Any]:
            return {
                "name": "VLM AutoEval Robot Policy",
                "model": self.model,
                "status": "running",
                "timestamp": datetime.now().isoformat(),
            }

        # Add the main action endpoint
        app.post("/act")(self.predict_action)

        # Add a reset endpoint
        @app.post("/reset")
        async def reset() -> dict[str, Any]:
            logger.info("Resetting history")
            self.history = (
                None if self.history_length is None else deque(maxlen=self.history_length)
            )
            return {"status": "reset successful"}

        # Add a startup event to ensure rate limiter monitoring is integrated with FastAPI's event loop
        @app.on_event("startup")
        async def startup_event() -> None:
            """Run when the application starts."""
            # Ensure rate limiter monitoring uses this event loop
            if rate_limiter._monitor_running and rate_limiter._monitor_task is None:
                loop = asyncio.get_running_loop()
                rate_limiter._monitor_task = loop.create_task(rate_limiter._monitor_loop())
                logger.info("Started rate limit monitoring in FastAPI's event loop")

        return app

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """
        Run the server

        Args:
            host: Host address
            port: Port number
        """
        self.app = self._create_app()

        # Configure server with increased timeout and request size limits
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            timeout_keep_alive=120,
            limit_concurrency=None,  # Remove concurrency limit entirely; rate limiter handles this
        )
        server = uvicorn.Server(config)

        server.run()


@dataclass
class DeployConfig:
    # Server Configuration
    host: str = "0.0.0.0"  # Host IP Address
    port: int = 8000  # Host Port
    model: str = "gemini/gemini-2.5-pro-preview-03-25"  # VLM model to use
    history_length: Optional[int] = None  # Optional history length (how many steps to remember)
    history_choice: Optional[Any] = (
        None  # Optional history choice (which index of history deque to use. Note that new steps to the deque are added to the end, so the oldest step is at index 0 and newest step is at index -1)
    )


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    """
    Deploy the VLM policy server

    Args:
        cfg: Deployment configuration
    """
    server = VLMPolicyServer(
        model=cfg.model, history_length=cfg.history_length, history_choice=cfg.history_choice
    )
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()
