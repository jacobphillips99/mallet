import asyncio
import base64
import copy
import logging
import traceback
from pathlib import Path

from vlm_autoeval_robot_benchmark.models.vlm import VLM, ImageInput, VLMRequest

logger = logging.getLogger(__name__)


# MODEL = "gpt-4o-mini"
MODEL = "gemini/gemini-2.0-flash"
# MODEL = "claude-3-5-sonnet-20240620"


async def test_vlm() -> None:
    # Initialize VLM
    vlm = VLM()

    # Test 1: Simple text request
    text_request = VLMRequest(
        prompt="What is 2+2?",
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

    # Test 2: Vision request with an image
    # First, let's create a simple test image path
    test_image_path = Path("assets/auto_eval_goal_images/put the eggplant in the blue sink.png")

    if test_image_path.exists():
        with open(test_image_path, "rb") as f:
            image_data = f.read()

        vision_request = VLMRequest(
            prompt="What's in this image?",
            model=MODEL,
            images=[
                ImageInput(
                    data=base64.b64encode(image_data).decode("utf-8"), mime_type="image/jpeg"
                )
            ],
        )

        many_vision_requests = [copy.deepcopy(vision_request) for _ in range(5)]
        for i, req in enumerate(many_vision_requests):
            req.prompt += f" ({i})"

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
    else:
        print("\nSkipping vision test - no test image found")


if __name__ == "__main__":
    asyncio.run(test_vlm())
