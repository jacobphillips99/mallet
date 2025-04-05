"""FastAPI server to handle requests from AutoEval."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from vlm_autoeval_robot_benchmark.models.vlm import VLM, ImageInput, VLMRequest

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="VLM AutoEval Robot Benchmark",
    description="API-driven VLM server for AutoEval robot benchmarking",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global VLM instance
vlm_instance = None


class RobotObservation(BaseModel):
    """Robot observation from AutoEval."""

    image: str  # Base64 encoded image
    proprio: List[float] = Field(default_factory=list)  # Proprioceptive states
    additional_info: Optional[Dict[str, Any]] = None


class RobotCommand(BaseModel):
    """Command to be sent to the robot."""

    actions: List[float] = Field(default_factory=list)
    success: bool = False
    info: Optional[Dict[str, Any]] = None


class VLMCommandRequest(BaseModel):
    """Request for VLM command generation."""

    observation: RobotObservation
    task_description: str
    history: Optional[List[Dict[str, Any]]] = None
    model: str = "gpt-4-vision-preview"
    system_prompt: Optional[str] = None


def translate_command_to_action(command: str) -> List[float]:
    """Translate a command from VLM into robot actions.

    Args:
        command: The command from the VLM

    Returns:
        The robot actions as a list of floats
    """
    # This is a simplified version based on the ECoT code from the README
    # In a real implementation, this would be more sophisticated

    command = command.lower()

    # Default action (no movement)
    action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Parse the command for movement directions
    # Format: [x, y, z, roll, pitch, yaw, gripper]

    # Forward/backward (x-axis)
    if "forward" in command:
        action[0] = 0.1
    elif "backward" in command:
        action[0] = -0.1

    # Left/right (y-axis)
    if "left" in command:
        action[1] = 0.1
    elif "right" in command:
        action[1] = -0.1

    # Up/down (z-axis)
    if "up" in command and "tilt" not in command:
        action[2] = 0.1
    elif "down" in command and "tilt" not in command:
        action[2] = -0.1

    # Tilt up/down (roll, around x-axis)
    if "tilt up" in command:
        action[3] = 0.1
    elif "tilt down" in command:
        action[3] = -0.1

    # Rotation (yaw, around z-axis)
    if "rotate counterclockwise" in command:
        action[5] = 0.1
    elif "rotate clockwise" in command:
        action[5] = -0.1

    # Gripper
    if "open gripper" in command:
        action[6] = 0.1
    elif "close gripper" in command:
        action[6] = -0.1

    # Return the normalized action
    return action


def get_default_system_prompt() -> str:
    """Get the default system prompt for VLM."""
    return """You are an expert robot control system. Your task is to control a robot arm to accomplish various tasks.

You will receive an image from the robot's camera and a description of the task to perform.
You should analyze the image and provide clear, concise instructions to move the robot.

Use commands like:
- "move forward/backward" (x-axis)
- "move left/right" (y-axis)
- "move up/down" (z-axis)
- "tilt up/down" (rotation around x-axis)
- "rotate clockwise/counterclockwise" (rotation around z-axis)
- "open gripper/close gripper" (control the gripper)
- "stop" (no movement)

You can combine commands, for example "move forward, rotate clockwise".
Keep your instructions simple and focused on one or two movements at a time.

If you believe the task is complete, say "TASK COMPLETE" at the end of your response.
"""


def init_vlm_instance() -> None:
    """Initialize the global VLM instance."""
    global vlm_instance

    if vlm_instance is None:
        vlm_instance = VLM()
        logger.info("VLM instance initialized")


@app.on_event("startup")
async def startup_event() -> None:
    """Startup event for the FastAPI app."""
    init_vlm_instance()


@app.get("/")
async def root() -> dict[str, Any]:
    """Root endpoint."""
    return {
        "name": "VLM AutoEval Robot Benchmark",
        "version": "0.1.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/health")
async def health() -> dict[str, Any]:
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/command")
async def generate_command(request: VLMCommandRequest) -> RobotCommand:
    """Generate a command for the robot based on the observation and task description.

    Args:
        request: The request containing the observation, task description, etc.

    Returns:
        The robot command
    """
    global vlm_instance
    init_vlm_instance()

    try:
        # Prepare the image
        image = ImageInput(data=request.observation.image, mime_type="image/jpeg")

        # Create a prompt using the task description
        prompt = f"Task: {request.task_description}\n\n"

        # Add history context if available
        if request.history:
            prompt += "Previous actions:\n"
            for entry in request.history[-5:]:  # Only show the last 5 entries
                if "command" in entry and "observation" in entry:
                    prompt += f"- Command: {entry['command']}\n"

            prompt += "\n"

        # Add proprioceptive state if available
        if request.observation.proprio:
            proprio_str = ", ".join([f"{val:.4f}" for val in request.observation.proprio])
            prompt += f"Current robot state: [{proprio_str}]\n\n"

        prompt += "Based on the image, what is the next action to take to complete the task?"

        # Create a VLM request
        vlm_request = VLMRequest(
            prompt=prompt,
            images=[image],
            model=request.model,
            system_prompt=request.system_prompt or get_default_system_prompt(),
            max_tokens=512,
            temperature=0.7,
        )

        # Generate a response
        if not vlm_instance:
            raise HTTPException(status_code=500, detail="VLM instance not initialized")

        response = await vlm_instance.generate(vlm_request)

        # Check if the task is complete
        task_complete = "TASK COMPLETE" in response.text.upper()

        # Translate the command to actions
        actions = translate_command_to_action(response.text)

        return RobotCommand(
            actions=actions,
            success=task_complete,
            info={
                "vlm_response": response.text,
                "model": response.model,
                "provider": response.provider,
                "response_ms": response.response_ms,
                "usage": response.usage,
            },
        )

    except Exception as e:
        logger.error(f"Error generating command: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def run_server(host: str = "0.0.0.0", port: int = 8000, log_level: str = "info") -> None:
    """Run the FastAPI server.

    Args:
        host: The host to bind to
        port: The port to bind to
        log_level: The log level
    """
    uvicorn.run(
        "vlm_autoeval_robot_benchmark.server:app", host=host, port=port, log_level=log_level
    )


if __name__ == "__main__":
    run_server()
