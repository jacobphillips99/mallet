#!/usr/bin/env python3
"""Test client for VLM AutoEval Robot Benchmark server."""

from dataclasses import dataclass
from urllib.parse import urlparse

import draccus
import numpy as np
import requests
from PIL import Image

from test_utils import AUTO_EVAL_TEST_UTILS


def test_server(
    task: str,
    host: str,
    port: int,
) -> bool:
    """Test the server with a simple request."""
    # Generate test image and proprioceptive data
    image = np.array(Image.open(AUTO_EVAL_TEST_UTILS[task]["start_image"]))
    proprio = np.random.rand(7).tolist()
    proprio[-1] = 1  # gripper is open
    instruction = AUTO_EVAL_TEST_UTILS[task]["task_instruction"]

    # Create payload
    payload = {
        "image": image.tolist(),  # Convert to list for JSON serialization
        "instruction": instruction,
        "proprio": proprio,
        "test": True,
    }

    # Send request
    try:
        print(f"Sending request with instruction: '{instruction}'")

        # Check if host is a full URL (like Modal endpoint) or just a hostname
        parsed_url = urlparse(host)
        if parsed_url.scheme:
            # If host is a full URL (like Modal endpoint), use it directly
            url = host
        else:
            # If host is just a hostname (like localhost), construct the full URL
            url = f"http://{host}:{port}/act"

        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})

        # Check response
        if response.status_code == 200:
            result = response.json()
            print("\nServer Response:")
            print(f"Action: {result.get('action')}")
            print(f"VLM Response: {result.get('vlm_response')}")
            return True
        else:
            print(f"Error: Server returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"Connection error: Could not connect to server at {url}")
        print("Make sure the server is running and accessible.")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


def test_health(host: str = "localhost", port: int = 8000) -> bool:
    """Test the health endpoint."""
    try:
        # Check if host is a full URL or just a hostname
        parsed_url = urlparse(host)
        if parsed_url.scheme:
            # If host is a full URL, append /health
            url = f"{host}/health" if not host.endswith("/health") else host
        else:
            # If host is just a hostname, construct the full URL
            url = f"http://{host}:{port}/health"

        print(f"Testing health endpoint at {url}")
        response = requests.get(url)
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


@dataclass
class TestConfig:
    # Server Configuration
    # For local testing: host="localhost", port=8000
    # For Modal endpoint: host="https://jacobphillips99--vlm-robot-policy-server-act.modal.run"
    host: str = "https://jacobphillips99--vlm-robot-policy-server-act.modal.run"
    port: int = 8000  # Only used when host is a hostname without scheme
    task: str = "eggplant_in_blue_sink"  # Task to test
    health_check: bool = False  # Whether to run health check


@draccus.wrap()
def run_tests(cfg: TestConfig) -> None:
    """
    Test the VLM Policy Server with the specified configuration

    Args:
        cfg: Test configuration
    """
    # Print test info
    parsed_url = urlparse(cfg.host)
    if parsed_url.scheme:
        print(f"Testing VLM Policy Server at {cfg.host}")
    else:
        print(f"Testing VLM Policy Server at http://{cfg.host}:{cfg.port}")
    print("=" * 50)

    # Test health endpoint
    if cfg.health_check:
        health_ok = test_health(cfg.host, cfg.port)
        print(f"Health check: {'OK' if health_ok else 'FAILED'}")
        print("\n" + "=" * 50 + "\n")

    # Test action endpoint
    test_server(cfg.task, cfg.host, cfg.port)

    print("\n" + "=" * 50)
    print("Tests completed!")


if __name__ == "__main__":
    run_tests()
