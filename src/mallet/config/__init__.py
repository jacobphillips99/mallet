"""Configuration utilities for VLM AutoEval Robot Benchmark."""

import logging
import os
from typing import Any, Dict

import yaml

from mallet.models.rate_limit import ProviderRateLimits, RateLimitConfig

logger = logging.getLogger(__name__)


def load_yaml_config(file_path: str) -> dict[str, Any]:
    """Load YAML configuration file."""
    try:
        with open(file_path, "r") as f:
            config = yaml.safe_load(f)
        logger.debug(f"Loaded config from {file_path}")
        return config or {}
    except Exception as e:
        logger.warning(f"Error loading config from {file_path}: {str(e)}")
        return {}


def get_config_path(filename: str) -> str | None:
    """
    Get the path to a neighboring configuration file.
    """
    # Check in current directory
    if os.path.exists(filename):
        return filename

    return None


def load_rate_limits() -> RateLimitConfig:
    """Load rate limits from YAML configuration.

    Returns:
        A RateLimitConfig object containing rate limits for all providers and models.
        If no config file is found, returns an empty config.
    """
    config_path = get_config_path("rate_limits.yaml")
    if config_path:
        yaml_data = load_yaml_config(config_path)
        # Convert the flat structure to our nested structure
        providers = {
            provider: ProviderRateLimits(models=models) for provider, models in yaml_data.items()
        }
        return RateLimitConfig(providers=providers)

    # If no config file is found, return an empty config
    return RateLimitConfig()
