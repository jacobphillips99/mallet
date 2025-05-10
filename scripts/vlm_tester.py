import asyncio
import base64
import copy
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path

import draccus
from test_utils import AUTO_EVAL_TEST_UTILS

from mallet.models.vlm import VLM, ImageInput, VLMHistory, VLMInput, VLMRequest

logger = logging.getLogger(__name__)


@dataclass
class TestConfig:
    # VLM Configuration
    model: str = "gpt-4o-mini"
    task: str = "eggplant_in_blue_sink"
    parallel_requests: int = 1
    include_history: bool = True


async def test_vlm(cfg: TestConfig) -> None:
    # Initialize VLM
    vlm = VLM()

    # Test 1: Simple text request
    text_request = VLMRequest(
        vlm_input=VLMInput(
            prompt="What is 2+2?",
        ),
        model=cfg.model,
    )

    print("\n=== Testing text-only request ===")
    try:
        response = await vlm.generate(text_request)
        print(f"Model: {response.model}")
        print(f"Response: {response.text}")
        print(f"Response time: {response.response_ms}ms")
        print(f"Token usage: {response.usage}")
    except Exception as e:
        print(f"Text request failed: {e}")

    task = cfg.task
    history_image_path = Path(AUTO_EVAL_TEST_UTILS[task]["start_image"])
    image_path = Path(AUTO_EVAL_TEST_UTILS[task]["goal_image"])
    task_instructions = AUTO_EVAL_TEST_UTILS[task]["task_instruction"]
    if not image_path.exists():
        logger.warning(f"Image path {image_path} does not exist")
        return
    if not history_image_path.exists():
        logger.warning(f"History image path {history_image_path} does not exist")
        return

    with open(image_path, "rb") as f:
        image_data = f.read()
    with open(history_image_path, "rb") as f:
        history_image_data = f.read()

    history_image_input = ImageInput(
        data=base64.b64encode(history_image_data).decode("utf-8"), mime_type="image/png"
    )
    if cfg.include_history:
        history = VLMHistory(
            prefix="This shows the history of a robotics episode.",
            vlm_inputs=[
                VLMInput(
                    prompt="This image shows the initial state of the scene.",
                    images=[history_image_input],
                )
            ],
            suffix="Use this history to answer the question below.",
            placement="before",
        )
    else:
        history = None

    vision_request = VLMRequest(
        vlm_input=VLMInput(
            prompt=f"The image below shows the end state of the scene. What actions would the robot have to take to reach this end state? The task instructions are: {task_instructions}",
            images=[
                ImageInput(data=base64.b64encode(image_data).decode("utf-8"), mime_type="image/png")
            ],
        ),
        model=cfg.model,
        history=history,
    )

    many_vision_requests = [copy.deepcopy(vision_request) for _ in range(cfg.parallel_requests)]
    for i, req in enumerate(many_vision_requests):
        req.vlm_input.prompt += f" ({i})"

    print("\n=== Testing vision request ===")
    responses = await vlm.generate_parallel(many_vision_requests)
    for i, maybe_response, maybe_exception in responses:
        try:
            if maybe_response is not None:
                response = maybe_response
                print(f"Response {i}:")
                print(f"Model: {response.model}")
                print(f"Response: {response.text}")
                print(f"Response time: {response.response_ms}ms")
                print(f"Token usage: {response.usage}")
            else:
                print(f"Vision request {i} failed: {maybe_exception}")
        except Exception as e:
            logger.error(f"Vision request {i} failed: {e}; traceback: {traceback.format_exc()}")


@draccus.wrap()
def run_tests(cfg: TestConfig) -> None:
    """
    Run VLM tests with the specified configuration

    Args:
        cfg: Test configuration
    """
    asyncio.run(test_vlm(cfg))


if __name__ == "__main__":
    run_tests()
