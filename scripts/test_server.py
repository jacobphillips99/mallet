"""
Simple test script to validate the VLM Policy Server.

Can make requests to a local server or a remote server via Modal.

Conforms to the `AutoEval` format for robot policy evaluation, which means
accepting a payload on /act and returning a 7-unit action vector.
"""

import traceback
from dataclasses import dataclass
from typing import Any

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
    host_str = host.replace("http://", "").replace("https://", "")
    port_str = f":{port}" if int(port) >= 0 else ""
    endpoint_str = f"/{endpoint.lstrip('/')}" if endpoint else ""
    return f"{protocol}{host_str}{port_str}{endpoint_str}"


def submit_request(url: str, payload: dict[str, Any]) -> bool:
    try:
        print("=" * 50)
        print(
            f"Sending {payload.get('test', '')} request with instruction: '{payload['instruction']}'"
        )
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 200:
            result = response.json()
            # VLA policies return a list of actions; repackage for consistency
            if isinstance(result, list):
                result = {"action": result}
            print("\nServer Response:")
            if payload.get("test"):
                print(f"Action: {result.get('action')}")
                print(f"VLM Response: {result.get('vlm_response')}")
            else:
                print(result)
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
        print(f"Error: {str(e)}; {traceback.format_exc()}")
        return False


def test_server(
    task: str,
    host: str,
    port: int,
    test: bool,
) -> None:
    """Test the server with a simple request."""
    url = get_url(host, port, "/act")
    print(f"Testing server at {url}")

    # Generate test image and proprioceptive data
    image = np.array(Image.open(AUTO_EVAL_TEST_UTILS[task]["start_image"]))
    proprio = np.random.rand(7).tolist()
    proprio[-1] = 1  # set gripper to open
    instruction = AUTO_EVAL_TEST_UTILS[task]["task_instruction"]

    # Create payload
    payload = {
        "image": image.tolist(),
        "instruction": instruction,
        "proprio": proprio,
    }
    if test:
        test_payload = {
            "test": True,
            **payload,
        }
        submit_request(url, test_payload)
    submit_request(url, payload)


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
        print(f"Health check error: {str(e)}; {traceback.format_exc()}")
        return False


@dataclass
class TestConfig:
    host: str = "localhost"
    port: int = 8000
    # host: str = "your-app.modal.run"
    # port: int = -1  # Server port for remote testing
    task: str = "eggplant_in_blue_sink"
    health_check: bool = True
    test_check: bool = True  # note that VLA servers do not support test requests


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
    test_server(cfg.task, cfg.host, cfg.port, cfg.test_check)

    print("\n" + "=" * 50)
    print("Tests completed!")


if __name__ == "__main__":
    run_tests()
