import asyncio

import json_numpy
import numpy as np
from fastapi import requests

from vlm_autoeval_robot_benchmark.servers.server import VLMPolicyServer

json_numpy.patch()
from vlm_autoeval_robot_benchmark.models.vlm import (
    VLM,
    ImageInput,
    VLMInput,
    VLMRequest,
    parse_vlm_response,
)

test_img = np.zeros((2, 2, 3), dtype=np.uint8)
test_img_2 = np.ones((2, 2, 3), dtype=np.uint8)

payload = {
    "image": test_img,
    "instruction": "this is an instruction",
    "proprio": None,
}

payload_2 = {
    "image": test_img_2,
    "instruction": "this is an instruction",
    "proprio": None,
}


async def main():
    server = VLMPolicyServer(model="gpt-4o-mini", history_length=1)
    request = server.prep_request(
        image=payload["image"], instruction=payload["instruction"], raw_proprio=payload["proprio"]
    )
    messages = server.vlm._prepare_messages(request)
    response = await server.vlm.generate(request)
    description, move_dict = parse_vlm_response(response.text)
    server.add_to_history(request, description, move_dict)

    breakpoint()
    request_2 = server.prep_request(
        image=payload_2["image"],
        instruction=payload_2["instruction"],
        raw_proprio=payload_2["proprio"],
    )
    messages_2 = server.vlm._prepare_messages(request_2)
    response_2 = await server.vlm.generate(request_2)
    description_2, move_dict_2 = parse_vlm_response(response_2.text)
    server.add_to_history(request_2, description_2, move_dict_2)
    breakpoint()


if __name__ == "__main__":
    #     host = "localhost"
    #     port = 8000
    #     url = f"http://{host}:{port}/act"
    #     response = requests.post(url, json=payload)
    #     print(response.json())
    asyncio.run(main())
