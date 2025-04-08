import asyncio
import base64
import copy
import logging
import traceback
from pathlib import Path

from vlm_autoeval_robot_benchmark.models.vlm import (
    VLM,
    ImageInput,
    VLMHistory,
    VLMInput,
    VLMRequest,
)

logger = logging.getLogger(__name__)


# MODEL = "gpt-4o-mini"
MODEL = "gemini/gemini-2.0-flash"
# MODEL = "claude-3-5-sonnet-20240620"


async def test_vlm() -> None:
    # Initialize VLM
    vlm = VLM()

    # Test 1: Simple text request
    text_request = VLMRequest(
        vlm_input=VLMInput(
            prompt="What is 2+2?",
        ),
        model=MODEL,
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

    # Test 2: Vision request with an image and history
    # goal image of put the eggplant in the blue sink = eggplant in blue sink
    test_image_path = Path("assets/auto_eval_goal_images/put the eggplant in the blue sink.png")
    # goal image of put the eggplant in the yellow basket = eggplant in yellow basket
    history_image_path = Path(
        "assets/auto_eval_goal_images/put the eggplant in the yellow basket.png"
    )

    with open(test_image_path, "rb") as f:
        image_data = f.read()
    with open(history_image_path, "rb") as f:
        history_image_data = f.read()

    history_image_input = ImageInput(
        data=base64.b64encode(history_image_data).decode("utf-8"), mime_type="image/jpeg"
    )
    history = VLMHistory(
        prompt="This shows the history of a robotics episode.",
        vlm_inputs=[
            VLMInput(
                prompt="This image shows the initial state of the scene.",
                images=[history_image_input],
            )
        ],
        placement="before",
    )

    vision_request = VLMRequest(
        vlm_input=VLMInput(
            prompt="The image below shows the end state of the scene. What changed from the historical image to the one below?",
            images=[
                ImageInput(
                    data=base64.b64encode(image_data).decode("utf-8"), mime_type="image/jpeg"
                )
            ],
        ),
        model=MODEL,
        history=history,
    )

    many_vision_requests = [copy.deepcopy(vision_request) for _ in range(1)]
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


if __name__ == "__main__":
    asyncio.run(test_vlm())
