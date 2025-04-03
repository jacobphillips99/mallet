import copy
import traceback
from vlm_autoeval_robot_benchmark.utils.rate_limit import rate_limiter
from vlm_autoeval_robot_benchmark.models.vlm import VLM
import asyncio
from pathlib import Path
import base64
from vlm_autoeval_robot_benchmark.models.vlm import VLMRequest, ImageInput

MODEL = "gpt-4o-mini"

async def test_vlm():
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
    test_image_path = Path("assets/put the eggplant in the blue sink.png")  
    
    if test_image_path.exists():
        with open(test_image_path, "rb") as f:
            image_data = f.read()
            
        vision_request = VLMRequest(
            prompt="What's in this image?",
            model=MODEL,
            images=[
                ImageInput(
                    data=base64.b64encode(image_data).decode('utf-8'),
                    mime_type="image/jpeg"
                )
            ]
        )

        many_vision_requests = [copy.deepcopy(vision_request) for _ in range(5)]
        for i, req in enumerate(many_vision_requests):
            req.prompt += f" ({i})"
        
        print("\n=== Testing vision request ===")
        responses = await vlm.generate_parallel(many_vision_requests)
        for i, response, maybe_exception in responses:
            try:
                if maybe_exception:
                    print(f"Vision request {i} failed: {maybe_exception}")
                else:
                    print(f"Response {i}:")
                    print(f"Model: {response.model}")
                    print(f"Response: {response.text}")
                    print(f"Response time: {response.response_ms}ms")
                    print(f"Token usage: {response.usage}")
            except Exception as e:
                print(f"Vision request failed: {e}; {traceback.format_exc()}")
    else:
        print("\nSkipping vision test - no test image found")

if __name__ == "__main__":
    asyncio.run(test_vlm())