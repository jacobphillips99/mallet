"""Rate limiting utilities for VLM API calls."""

import time
import threading
import asyncio
from typing import Dict, Optional
import logging
import typing as t
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class ModelRateLimitConfig(BaseModel):
    """Rate limit configuration for a specific model."""
    
    requests_per_minute: int = Field(default=0, ge=0)
    tokens_per_minute: int = Field(default=0, ge=0)
    concurrent_requests: int = Field(default=10, gt=0)
    
    class Config:
        """Pydantic config."""
        extra = "ignore"


class ProviderRateLimits(BaseModel):
    """Rate limits for all models of a provider."""
    
    models: dict[str, ModelRateLimitConfig] = Field(default_factory=dict)


class RateLimitConfig(BaseModel):
    """Top level rate limit configuration for all providers."""
    
    providers: dict[str, ProviderRateLimits] = Field(default_factory=dict)


class RateLimit:
    """Rate limit handler for API calls."""
    
    def __init__(self):
        """Initialize the rate limit handler."""
        self._provider_model_configs: Dict[str, Dict[str, RateLimitConfig]] = {}
        self._provider_model_states: Dict[str, Dict[str, Dict]] = {}
        self._locks: Dict[str, Dict[str, threading.Lock]] = {}
        self._semaphores: Dict[str, Dict[str, asyncio.Semaphore]] = {}
    
    @property
    def providers(self) -> list[str]:
        """Get list of registered providers.
        
        Returns:
            List of provider names that have been registered
        """
        return list(self._provider_model_configs.keys())
    
    def register_model(self, provider: str, model: str, config: RateLimitConfig):
        """Register a model with its rate limit configuration.
        
        Args:
            provider: The provider name (e.g., "openai")
            model: The model name (e.g., "gpt-4o")
            config: Rate limit configuration
        """
        if provider not in self._provider_model_configs:
            self._provider_model_configs[provider] = {}
            self._provider_model_states[provider] = {}
            self._locks[provider] = {}
            self._semaphores[provider] = {}
            
        self._provider_model_configs[provider][model] = config
        
        # Initialize state
        self._provider_model_states[provider][model] = {
            "request_timestamps": [],
            "token_usage": [],
            "last_cleanup_time": time.time()
        }
        
        # Initialize lock
        self._locks[provider][model] = threading.Lock()
        
        # Initialize semaphore for concurrent request limiting
        self._semaphores[provider][model] = asyncio.Semaphore(config.concurrent_requests)
        
        logger.info(f"Registered model {provider}/{model} with config: {config}")
    
    def get_config(self, provider: str, model: str) -> Optional[RateLimitConfig]:
        """Get the rate limit configuration for a provider/model.
        
        Args:
            provider: The provider name
            model: The model name
            
        Returns:
            The rate limit configuration if it exists, None otherwise
        """
        if provider not in self._provider_model_configs:
            return None
        return self._provider_model_configs[provider].get(model)
    
    def _cleanup_old_data(self, provider: str, model: str):
        """Clean up old timestamps and token usage data.
        
        Args:
            provider: The provider name
            model: The model name
        """
        state = self._provider_model_states[provider][model]
        current_time = time.time()
        
        # Clean up only once every few seconds to avoid excessive cleaning
        if current_time - state["last_cleanup_time"] < 5:
            return
            
        one_minute_ago = current_time - 60
        
        # Filter out timestamps older than one minute
        state["request_timestamps"] = [
            ts for ts in state["request_timestamps"] if ts > one_minute_ago
        ]
        
        # Filter out token usage older than one minute
        state["token_usage"] = [
            (ts, tokens) for ts, tokens in state["token_usage"] 
            if ts > one_minute_ago
        ]
        
        state["last_cleanup_time"] = current_time
    
    async def acquire(self, provider: str, model: str, tokens: int = 0) -> bool:
        """Acquire permission to make an API call.
        
        Args:
            provider: The provider name
            model: The model name
            tokens: Estimated token usage for this request
            
        Returns:
            True if permission is granted, False otherwise
        """
        config = self.get_config(provider, model)
        if not config:
            logger.warning(f"No rate limit config for {provider}/{model}, allowing request")
            return True
            
        # Wait for semaphore (limits concurrent requests)
        async with self._semaphores[provider][model]:
            with self._locks[provider][model]:
                self._cleanup_old_data(provider, model)
                
                state = self._provider_model_states[provider][model]
                current_time = time.time()
                
                # Check requests per minute
                if config.requests_per_minute > 0:
                    current_rpm = len(state["request_timestamps"])
                    if current_rpm >= config.requests_per_minute:
                        logger.warning(
                            f"Rate limit exceeded for {provider}/{model}: "
                            f"{current_rpm}/{config.requests_per_minute} requests per minute"
                        )
                        return False
                
                # Check tokens per minute
                if config.tokens_per_minute > 0 and tokens > 0:
                    current_tpm = sum(t for _, t in state["token_usage"])
                    if current_tpm + tokens > config.tokens_per_minute:
                        logger.warning(
                            f"Token rate limit exceeded for {provider}/{model}: "
                            f"{current_tpm + tokens}/{config.tokens_per_minute} tokens per minute"
                        )
                        return False
                
                # Update state with this request
                state["request_timestamps"].append(current_time)
                if tokens > 0:
                    state["token_usage"].append((current_time, tokens))
                
                return True
    
    async def wait_and_acquire(self, provider: str, model: str, tokens: int = 0, max_retries: int = 10):
        """Wait until rate limit allows and acquire permission.
        
        Args:
            provider: The provider name
            model: The model name
            tokens: Estimated token usage for this request
            max_retries: Maximum number of retry attempts
            
        Returns:
            True if permission is granted, raises an exception if max retries exceeded
        """
        for attempt in range(max_retries):
            if await self.acquire(provider, model, tokens):
                return True
                
            # Exponential backoff with jitter
            backoff_time = min(2 ** attempt + (0.1 * attempt), 60)
            logger.info(f"Rate limit reached, waiting {backoff_time:.2f}s before retry")
            await asyncio.sleep(backoff_time)
        
        raise RuntimeError(f"Failed to acquire rate limit after {max_retries} retries")
    
    def record_usage(self, provider: str, model: str, tokens_used: int):
        """Record actual token usage after a request.
        
        This is used to update token usage tracking with actual values rather than estimates.
        
        Args:
            provider: The provider name
            model: The model name
            tokens_used: Actual tokens used in the request
        """
        if not self.get_config(provider, model):
            return
            
        with self._locks[provider][model]:
            state = self._provider_model_states[provider][model]
            
            # Update the most recent token usage record
            if state["token_usage"]:
                timestamp, _ = state["token_usage"][-1]
                state["token_usage"][-1] = (timestamp, tokens_used)
    
    def get_usage_stats(self, provider: str, model: str):
        """Get current usage statistics.
        
        Args:
            provider: The provider name
            model: The model name
            
        Returns:
            Dictionary with current usage statistics
        """
        if not self.get_config(provider, model):
            return {"error": "Model not registered"}
            
        with self._locks[provider][model]:
            self._cleanup_old_data(provider, model)
            state = self._provider_model_states[provider][model]
            
            config = self._provider_model_configs[provider][model]
            current_rpm = len(state["request_timestamps"])
            current_tpm = sum(t for _, t in state["token_usage"])
            
            return {
                "requests_per_minute": {
                    "current": current_rpm,
                    "limit": config.requests_per_minute,
                    "percent": (current_rpm / config.requests_per_minute * 100) if config.requests_per_minute else 0
                },
                "tokens_per_minute": {
                    "current": current_tpm,
                    "limit": config.tokens_per_minute,
                    "percent": (current_tpm / config.tokens_per_minute * 100) if config.tokens_per_minute else 0
                },
                "concurrent_requests": {
                    "current": config.concurrent_requests - self._semaphores[provider][model]._value,
                    "limit": config.concurrent_requests,
                    "percent": ((config.concurrent_requests - self._semaphores[provider][model]._value) / 
                               config.concurrent_requests * 100)
                }
            }


# Global rate limiter instance
rate_limiter = RateLimit() 