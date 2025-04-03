"""Configuration utilities for VLM AutoEval Robot Benchmark."""

import os
import yaml
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from pydantic import BaseModel, Field, root_validator

logger = logging.getLogger(__name__)


class ModelRateLimit(BaseModel):
    """Rate limit configuration for a specific model."""
    
    requests_per_minute: int = Field(default=0, ge=0)
    tokens_per_minute: int = Field(default=0, ge=0)
    concurrent_requests: int = Field(default=10, gt=0)
    
    class Config:
        """Pydantic config."""
        extra = "ignore"


class ProviderRateLimits(BaseModel):
    """Rate limit configuration for all models of a provider."""
    
    __root__: Dict[str, ModelRateLimit]
    
    def __iter__(self):
        return iter(self.__root__)
    
    def __getitem__(self, item):
        return self.__root__[item]
    
    def items(self):
        return self.__root__.items()


class RateLimitConfig(BaseModel):
    """Rate limit configuration for all providers."""
    
    __root__: Dict[str, Dict[str, ModelRateLimit]] = Field(default_factory=dict)
    
    def __iter__(self):
        return iter(self.__root__)
    
    def __getitem__(self, item):
        return self.__root__[item]
    
    def items(self):
        return self.__root__.items()


class ApiKeys(BaseModel):
    """API keys for different providers."""
    
    openai: Optional[str] = None
    anthropic: Optional[str] = None
    google: Optional[str] = None
    mistral: Optional[str] = None
    cohere: Optional[str] = None
    
    @root_validator(pre=True)
    def load_from_env(cls, values):
        """Load API keys from environment variables if not provided."""
        env_mapping = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "cohere": "COHERE_API_KEY",
        }
        
        for field, env_var in env_mapping.items():
            if field not in values or values[field] is None:
                env_value = os.environ.get(env_var)
                if env_value:
                    values[field] = env_value
        
        return values
    
    def as_dict(self) -> Dict[str, str]:
        """Convert to dictionary, filtering out None values."""
        return {k: v for k, v in self.dict().items() if v is not None}


def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """Load YAML configuration file.
    
    Args:
        file_path: Path to the YAML file
        
    Returns:
        Dictionary with configuration values
    """
    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.debug(f"Loaded config from {file_path}")
        return config or {}
    except Exception as e:
        logger.warning(f"Error loading config from {file_path}: {str(e)}")
        return {}


def get_config_path(filename: str) -> Optional[str]:
    """Get the path to a configuration file.
    
    Checks for the file in the following locations (in order):
    1. Current working directory
    2. ~/.vlm_autoeval/ directory
    3. Package config directory
    
    Args:
        filename: Name of the configuration file
        
    Returns:
        Path to the configuration file or None if not found
    """
    # Check in current directory
    if os.path.exists(filename):
        return filename
    
    # Check in ~/.vlm_autoeval/ directory
    home_dir = os.path.expanduser("~")
    home_config_path = os.path.join(home_dir, ".vlm_autoeval", filename)
    if os.path.exists(home_config_path):
        return home_config_path
    
    # Check in package config directory
    package_dir = os.path.dirname(__file__)
    package_config_path = os.path.join(package_dir, filename)
    if os.path.exists(package_config_path):
        return package_config_path
    
    return None


def get_api_keys() -> ApiKeys:
    """Get API keys configuration.
    
    Returns:
        ApiKeys object loaded from environment variables
    """
    return ApiKeys()


def load_rate_limits() -> RateLimitConfig:
    """Load rate limits from YAML configuration.
    
    Returns:
        RateLimitConfig object
    """
    config_path = get_config_path("rate_limits.yaml")
    if config_path:
        yaml_data = load_yaml_config(config_path)
        return RateLimitConfig.parse_obj(yaml_data)
    
    # If no config file is found, return an empty config
    return RateLimitConfig() 