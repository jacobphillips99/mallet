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
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict

import draccus
import json_numpy
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from vlm_autoeval_robot_benchmark.models.rate_limit import rate_limiter
from vlm_autoeval_robot_benchmark.models.translation import PromptTemplate, build_standard_prompt
from vlm_autoeval_robot_benchmark.models.vlm import (
    VLM,
    ImageInput,
    VLMInput,
    VLMRequest,
    parse_vlm_response,
)
from vlm_autoeval_robot_benchmark.utils.ecot_primitives.inverse_ecot_primitive_movements import (
    text_to_move_vector,
)
from vlm_autoeval_robot_benchmark.utils.robot_utils import GRIPPER_INDEX, get_gripper_position

json_numpy.patch()
# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
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
        => Returns  {"action": np.ndarray}
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        """
        Initialize the VLM-based policy server

        Args:
            model: The VLM model to use
        """
        # Initialize VLM instance
        self.vlm = VLM()
        self.model = model
        logger.info(f"VLM Policy Server initialized with model: {self.model}")

    async def predict_action(self, payload: Dict[str, Any]) -> JSONResponse:
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

            image = payload["image"]
            instruction = payload["instruction"]
            proprio = payload.get("proprio", None)  # proprio is optional
            history = payload.get("history", None)  # history is optional

            # Prepare the VLM request
            image_b64 = image_server_helper(image)
            vlm_image = ImageInput(data=image_b64, mime_type="image/png")
            gripper_position = (
                None if proprio is None else get_gripper_position(proprio[GRIPPER_INDEX])
            )

            # Create a prompt for the VLM
            prompt = build_standard_prompt(
                prompt_template=PromptTemplate(
                    env_desc="You are looking at a robotics environment.",
                    task_desc=instruction,
                    gripper_position=gripper_position,
                    history_flag=history is not None,
                )
            )

            # Create history context if available
            # vlm_history = None
            # if history:
            #     # Simplified history handling - would need to be expanded for real use
            #     pass

            # Create and send the VLM request
            vlm_request = VLMRequest(
                vlm_input=VLMInput(
                    prompt=prompt,
                    images=[vlm_image],
                ),
                model=self.model,
                history=None,
            )

            response = await self.vlm.generate(vlm_request)
            try:
                # Extract the JSON part of the response, convert to move vector
                description, move_dict = parse_vlm_response(response.text)
                action = text_to_move_vector(move_dict).tolist()
                logger.info(f"VLM description: {description}")

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
        async def health_check() -> Dict[str, Any]:
            return {
                "status": "healthy",
                "model": self.model,
                "timestamp": datetime.now().isoformat(),
            }

        @app.get("/")
        async def root() -> Dict[str, Any]:
            return {
                "name": "VLM AutoEval Robot Policy",
                "model": self.model,
                "status": "running",
                "timestamp": datetime.now().isoformat(),
            }

        # Add the main action endpoint
        app.post("/act")(self.predict_action)

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
            limit_concurrency=None,  # Remove concurrency limit entirely
        )
        server = uvicorn.Server(config)

        server.run()


@dataclass
class DeployConfig:
    # Server Configuration
    host: str = "0.0.0.0"  # Host IP Address
    port: int = 8000  # Host Port
    model: str = "gemini/gemini-2.5-pro-preview-03-25"  # VLM model to use


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    """
    Deploy the VLM policy server

    Args:
        cfg: Deployment configuration
    """
    server = VLMPolicyServer(model=cfg.model)
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()
