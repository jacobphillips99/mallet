#!/usr/bin/env python3
"""Test client for VLM AutoEval Robot Benchmark server."""

import argparse
from typing import Tuple

import numpy as np
import requests


def generate_test_image(size: Tuple[int, int, int] = (256, 256, 3)) -> np.ndarray:
    """Generate a simple test image."""
    # Create a simple gradient image
    img = np.zeros(size, dtype=np.uint8)
    # Add a red square
    img[50:200, 50:200, 0] = 255
    # Add a blue circle
    y, x = np.ogrid[0 : size[0], 0 : size[1]]
    mask = (x - size[1] // 2) ** 2 + (y - size[0] // 2) ** 2 <= (50**2)
    img[mask, 2] = 255
    return img


def test_server(
    host: str = "localhost",
    port: int = 8000,
    instruction: str = "Pick up the red cube and place it in the blue box.",
) -> bool:
    """Test the server with a simple request."""
    print(f"Testing server at http://{host}:{port}/act")

    # Generate test image and proprioceptive data
    image = generate_test_image()
    proprio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    # Create payload
    payload = {
        "image": image.tolist(),  # Convert to list for JSON serialization
        "instruction": instruction,
        "proprio": proprio,
    }

    # Send request
    try:
        print(f"Sending request with instruction: '{instruction}'")
        response = requests.post(
            f"http://{host}:{port}/act", json=payload, headers={"Content-Type": "application/json"}
        )

        # Check response
        if response.status_code == 200:
            result = response.json()
            print("\nServer Response:")
            print(f"Action: {result.get('action')}")
            print(f"Success: {result.get('success')}")

            # Print VLM response if available
            if result.get("info", {}).get("vlm_response"):
                print("\nVLM Response (truncated):")
                vlm_response = result["info"]["vlm_response"]
                # Print first 300 chars to avoid too much output
                print(f"{vlm_response[:300]}..." if len(vlm_response) > 300 else vlm_response)

            # Print model info if available
            if result.get("info", {}).get("model"):
                print(f"\nModel: {result['info']['model']}")
                print(f"Provider: {result['info']['provider']}")
                print(f"Response time: {result['info']['response_ms']} ms")

            return True
        else:
            print(f"Error: Server returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"Connection error: Could not connect to server at {host}:{port}")
        print("Make sure the server is running and accessible.")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


def test_health(host: str = "localhost", port: int = 8000) -> bool:
    """Test the health endpoint."""
    try:
        print(f"Testing health endpoint at http://{host}:{port}/health")
        response = requests.get(f"http://{host}:{port}/health")
        if response.status_code == 200:
            print("Health check successful!")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"Health check failed with status code {response.status_code}")
            return False
    except Exception as e:
        print(f"Health check error: {str(e)}")
        return False


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test the VLM Policy Server")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument(
        "--instruction",
        type=str,
        default="Pick up the red block and place it in the blue box.",
        help="Task instruction to send to the server",
    )
    args = parser.parse_args()

    # Print test info
    print(f"Testing VLM Policy Server at {args.host}:{args.port}")
    print("=" * 50)

    # Test health endpoint
    health_ok = test_health(args.host, args.port)

    print("\n" + "=" * 50 + "\n")

    # Test action endpoint
    if health_ok:
        test_server(args.host, args.port, args.instruction)

    print("\n" + "=" * 50)
    print("Tests completed!")
