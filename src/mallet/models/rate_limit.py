"""Rate limiting utilities for VLM API calls with built-in monitoring."""

import asyncio
import json
import logging
import os
import threading
import time
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

RATE_LIMIT_STATS_PATH = "/tmp/rate_limit_stats.json"


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
    """Rate limit handler for API calls with built-in monitoring."""

    def __init__(
        self,
        stats_path: str = RATE_LIMIT_STATS_PATH,
        monitor_interval: float = 1.0,
        disable_monitoring: bool = False,
    ) -> None:
        """Initialize the rate limit handler.

        Args:
            stats_path: Path to write stats to. If None, uses /tmp/rate_limit_stats.json
            monitor_interval: How often to update the stats file (seconds)
            disable_monitoring: If True, won't write stats to file
        """
        self._provider_model_configs: dict[str, dict[str, ModelRateLimitConfig]] = {}
        self._provider_model_states: dict[str, dict[str, dict]] = {}
        self._locks: dict[str, dict[str, threading.Lock]] = {}
        self._semaphores: dict[str, dict[str, asyncio.Semaphore]] = {}

        # Stats monitoring
        self._stats_path = stats_path
        self._monitor_interval = monitor_interval
        self._monitor_running = not disable_monitoring
        self._monitor_task: Optional[asyncio.Task] = None

        # Start monitoring automatically if not disabled
        if self._monitor_running:
            # This will be initialized when register_model is first called
            # or when get_usage_stats is first called
            pass

    def _ensure_monitor_started(self) -> None:
        """Ensure the monitoring task is started if enabled."""
        if self._monitor_running and self._monitor_task is None:
            try:
                # Get the current event loop or create one if needed
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                # Create the monitoring task
                self._monitor_task = loop.create_task(self._monitor_loop())
                logger.info(
                    f"Started rate limit monitoring, writing to {self._stats_path} every {self._monitor_interval}s"
                )
            except Exception as e:
                logger.error(f"Failed to start monitoring: {e}")
                self._monitor_running = False

    @property
    def providers(self) -> list[str]:
        """Get list of registered providers.

        Returns:
            List of provider names that have been registered
        """
        return list(self._provider_model_configs.keys())

    def register_model(self, provider: str, model: str, config: ModelRateLimitConfig) -> None:
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
            "last_cleanup_time": time.time(),
        }

        # Initialize lock
        self._locks[provider][model] = threading.Lock()

        # Initialize semaphore for concurrent request limiting
        self._semaphores[provider][model] = asyncio.Semaphore(config.concurrent_requests)

        # Start monitoring if enabled and not already started
        self._ensure_monitor_started()
        logger.info(f"Registered model {provider}/{model} with rate limit config {config}")

    def get_config(self, provider: str, model: str) -> Optional[ModelRateLimitConfig]:
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

    def _cleanup_old_data(self, provider: str, model: str) -> None:
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
            (ts, tokens) for ts, tokens in state["token_usage"] if ts > one_minute_ago
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
                    current_tpm = sum(tok for _, tok in state["token_usage"])
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

    async def wait_and_acquire(
        self, provider: str, model: str, tokens: int = 0, max_retries: int = 10
    ) -> bool:
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
            backoff_time = min(2**attempt + (0.1 * attempt), 60)
            logger.info(f"Rate limit reached, waiting {backoff_time:.2f}s before retry")
            await asyncio.sleep(backoff_time)

        raise RuntimeError(f"Failed to acquire rate limit after {max_retries} retries")

    def record_usage(self, provider: str, model: str, tokens_used: int) -> None:
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

    def get_usage_stats(
        self, provider: Optional[str] = None, model: Optional[str] = None
    ) -> dict[str, Any]:
        """Get current usage statistics.

        Args:
            provider: The provider name (optional, if None returns stats for all providers)
            model: The model name (optional, if None returns stats for all models of the provider)

        Returns:
            Dictionary with current usage statistics
        """
        # Start monitoring if enabled and not already started
        self._ensure_monitor_started()

        result: dict[str, Any] = {}

        if provider is None:
            # Return stats for all providers
            for provider_name in self.providers:
                result[provider_name] = self.get_usage_stats(provider_name)
            return result

        if provider not in self._provider_model_configs:
            return {"error": "Provider not registered"}

        if model is None:
            # Return stats for all models of this provider
            provider_stats = {}
            for model_name in self._provider_model_configs[provider]:
                provider_stats[model_name] = self.get_usage_stats(provider, model_name)
            return provider_stats

        if model not in self._provider_model_configs[provider]:
            return {"error": "Model not registered"}

        with self._locks[provider][model]:
            self._cleanup_old_data(provider, model)
            state = self._provider_model_states[provider][model]

            config = self._provider_model_configs[provider][model]
            current_rpm = len(state["request_timestamps"])
            current_tpm = sum(tok for _, tok in state["token_usage"])

            return {
                "requests_per_minute": {
                    "current": current_rpm,
                    "limit": config.requests_per_minute,
                    "percent": (
                        (current_rpm / config.requests_per_minute * 100)
                        if config.requests_per_minute
                        else 0
                    ),
                },
                "tokens_per_minute": {
                    "current": current_tpm,
                    "limit": config.tokens_per_minute,
                    "percent": (
                        (current_tpm / config.tokens_per_minute * 100)
                        if config.tokens_per_minute
                        else 0
                    ),
                },
                "concurrent_requests": {
                    "current": config.concurrent_requests
                    - self._semaphores[provider][model]._value,
                    "limit": config.concurrent_requests,
                    "percent": (
                        (config.concurrent_requests - self._semaphores[provider][model]._value)
                        / config.concurrent_requests
                        * 100
                    ),
                },
            }

    async def _monitor_loop(self) -> None:
        """Background task to periodically write stats to file."""
        try:
            while self._monitor_running:
                stats = {"timestamp": time.time(), "stats": self.get_usage_stats()}

                # Ensure directory exists
                os.makedirs(os.path.dirname(self._stats_path), exist_ok=True)

                # Write stats to file atomically
                tmp_path = f"{self._stats_path}.tmp"
                with open(tmp_path, "w") as f:
                    json.dump(stats, f, indent=2)
                os.rename(tmp_path, self._stats_path)

                await asyncio.sleep(self._monitor_interval)
        except Exception as e:
            logger.error(f"Error in monitor loop: {e}")
            self._monitor_running = False


# Global rate limiter instance
rate_limiter = RateLimit()
