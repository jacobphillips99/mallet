"""
VLM AutoEval Robot Benchmark

A package to create an API-driven VLM server to tunnel requests/responses
to the real-world robot evaluation project AutoEval.
"""

__version__ = "0.1.0"

import os
import logging

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)