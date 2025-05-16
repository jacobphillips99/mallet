"""
Simple script to ping a VLA server to keep it alive; important for GPU servers that have a long spin-up time.
Note that the script will exit after the number of keep-alive checks specified by `interval_multiple_limit`.

Reccomended usage is to run this script while queuing an evaluation on AutoEval, then killing it once the job has begun.

Usage:

python -m scripts.keep_alive --host <host> --port <port> --interval <interval> --interval_multiple_limit <interval_multiple_limit>
"""

import time
import traceback
from dataclasses import dataclass

import draccus
import requests

from mallet.utils import get_url
from modal_servers.vla import DEFAULT_SCALEDOWN_WINDOW

ENDPOINT = "/health"


def keep_alive(host: str, port: int, interval: float, interval_multiple_limit: int) -> None:
    url = get_url(host, port, ENDPOINT)
    tic = time.time()

    for _ in range(interval_multiple_limit):
        try:
            inner_tic = time.time()
            print(f"Testing health endpoint at {url}")
            response = requests.get(url)
            inner_time = time.time() - inner_tic
            if response.status_code == 200:
                print(f"Health check successful in {round(inner_time, 2)} seconds!")
                print(f"Response: {response.json()}")
            else:
                print(
                    f"Health check failed with status code {response.status_code} in {round(inner_time, 2)} seconds"
                )
        except Exception as e:
            print(f"Health check error: {str(e)}; {traceback.format_exc()}")
            raise e

        time_alive = time.time() - tic
        minutes, seconds = divmod(time_alive, 60)
        print(f"Kept {host}:{port} alive for {minutes:.0f} minutes and {seconds:.0f} seconds")
        time.sleep(interval * 0.9)

    print(f"Completed {interval_multiple_limit} keep-alive checks for {host}:{port}. Exiting.")


@dataclass
class KeepAliveConfig:
    host: str
    port: int
    interval: float = DEFAULT_SCALEDOWN_WINDOW
    interval_multiple_limit: int = (
        10  # only keep alive for interval * interval_multiple_limit seconds
    )


@draccus.wrap()
def main(cfg: KeepAliveConfig) -> None:
    keep_alive(cfg.host, cfg.port, cfg.interval, cfg.interval_multiple_limit)


if __name__ == "__main__":
    main()
