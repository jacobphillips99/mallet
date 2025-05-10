#!/usr/bin/env python3
"""Test client for VLM AutoEval Robot Benchmark server."""

from dataclasses import dataclass

import draccus
import numpy as np
import requests
from PIL import Image
from test_utils import AUTO_EVAL_TEST_UTILS


def get_url(host: str, port: int, endpoint: str | None = None, protocol: str = "http://") -> str:
    """
    Get the URL for a given host and port; if port is negative, skip it.
    Cleans the host and endpoint strings
    """
    # Remove http:// or https:// from host if present
    host_str = host.replace("http://", "").replace("https://", "")
    port_str = f":{port}" if int(port) >= 0 else ""
    endpoint_str = f"/{endpoint.lstrip('/')}" if endpoint else ""
    return f"{protocol}{host_str}{port_str}{endpoint_str}"


def test_server(
    task: str,
    host: str,
    port: int,
) -> bool:
    """Test the server with a simple request."""
    url = get_url(host, port, "/act")
    print(f"Testing server at {url}")

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
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
        )

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


def test_health(host: str, port: int) -> bool:
    """Test the health endpoint."""
    try:
        url = get_url(host, port, "/health")
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
    # local testing
    # host: str = "localhost"  # Server host
    # port: int = 8000  # Server port for local testing
    # remote testing
    # removing "s" for test --> seems to work!
    host: str = "jacobphillips99--vlm-robot-policy-server-fastapi-app.modal.run"
    port: int = -1  # Server port for remote testing
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
    print(f"Testing VLM Policy Server at host {cfg.host} and port {cfg.port}")
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
