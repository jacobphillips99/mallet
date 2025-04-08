"""VLM class for handling async parallel requests using litellm."""

import asyncio
import base64
import json
import logging
import os
import re
import time
import traceback
import typing as t
from typing import Any, Dict, List, Optional, Tuple, Union

import litellm
from pydantic import BaseModel, Field

from vlm_autoeval_robot_benchmark.config import RateLimitConfig, load_rate_limits
from vlm_autoeval_robot_benchmark.models.rate_limit import rate_limiter
from vlm_autoeval_robot_benchmark.models.translation import build_prompt

logger = logging.getLogger(__name__)


def get_provider_from_model(model: str) -> str:
    """Get the provider name from a model name.

    Args:
        model: The model name

    Returns:
        The provider name

    Raises:
        ValueError: If provider cannot be determined
    """
    # Fall back to heuristics
    if model.startswith("gpt") or "openai" in model:
        return "openai"
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("gemini"):
        return "gemini"
    raise ValueError(f"Could not determine provider for model: {model}")


class VLMResponse(BaseModel):
    """Response from a VLM model."""

    text: str
    model: str
    provider: str
    usage: Dict[str, Any] = Field(default_factory=dict)
    response_ms: int = 0
    raw_response: Optional[Dict[str, Any]] = None
    stop_reason: Optional[str] = None


class ImageInput(BaseModel):
    """Image input for VLM requests."""

    data: Union[str, bytes]  # Base64 encoded string or raw bytes
    mime_type: str = "image/jpeg"


class VLMInput(BaseModel):
    """Input content for a VLM request."""

    prompt: str
    images: Optional[List[ImageInput]] = None


class VLMHistory(BaseModel):
    prompt: str
    vlm_inputs: List[VLMInput]
    placement: str = "before"


class VLMRequest(BaseModel):
    """Request to a VLM model."""

    vlm_input: VLMInput
    max_tokens: int = 8192
    temperature: float = 0.0
    top_p: float = 0.9
    model: str = "gpt-4o-mini"
    provider: Optional[str] = None  # If None, will be derived from model
    system_prompt: Optional[str] = None
    stream: bool = False
    timeout: float = 120.0
    extra_params: Dict[str, Any] = Field(default_factory=dict)
    history: Optional[VLMHistory] = None  # List of previous inputs in the conversation


class VLM:
    """VLM class for handling async parallel requests using litellm."""

    def __init__(self, rate_limits: Optional[RateLimitConfig] = None) -> None:
        """Initialize the VLM class.

        Args:
            rate_limits: Rate limit configurations. If None, will try to load from yaml file.
        """
        self._setup_rate_limits(rate_limits)
        self._check_api_keys()

    def _check_api_keys(self) -> None:
        """Check if API keys are set for the available providers."""
        for provider in rate_limiter.providers:
            if f"{provider.upper()}_API_KEY" not in os.environ:
                raise ValueError(f"API key for {provider} is not set")

    async def _acheck_model_available(self, model: str) -> bool:
        vlm_request = VLMRequest(model=model, vlm_input=VLMInput(prompt="Hello, world!"))
        try:
            await self.generate(vlm_request)
            return True
        except Exception as e:
            logger.error(
                f"Error checking model availability for {model}: {str(e)}; {traceback.format_exc()}"
            )
            return False

    def check_model_available(self, model: str) -> bool:
        return asyncio.run(self._acheck_model_available(model))

    def _setup_rate_limits(self, rate_limits: Optional[RateLimitConfig]) -> None:
        """Set up rate limits for different providers/models.

        Args:
            rate_limits: Rate limit configurations. If None, will try to load from yaml file.
        """
        if rate_limits is None:
            # Try to load from YAML config
            rate_limits = load_rate_limits()

        # Register models with rate limiter
        for provider, provider_limits in rate_limits.providers.items():
            for model, model_limits in provider_limits.models.items():
                rate_limiter.register_model(provider, model, model_limits)

    @staticmethod
    def _prepare_images(images: List[ImageInput]) -> List[Dict[str, Any]]:
        """Prepare images for litellm API.

        Args:
            images: List of image inputs

        Returns:
            List of formatted image objects for litellm
        """
        formatted_images = []

        for img in images:
            # Ensure image is base64 encoded
            if isinstance(img.data, bytes):
                b64_data = base64.b64encode(img.data).decode("utf-8")
            else:
                # Assume it's already base64 encoded
                b64_data = img.data

            formatted_images.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{img.mime_type};base64,{b64_data}"},
                },
            )

        return formatted_images

    def _prepare_content(self, vlm_input: VLMInput) -> List[Dict[str, Any]]:
        if vlm_input.images:
            # Format for multi-modal models
            content = [{"type": "text", "text": vlm_input.prompt}]

            # interleave text breaks with images
            formatted_images = self._prepare_images(vlm_input.images)
            interleaved_content = [
                [{"type": "text", "text": f"Image {i+1}:"}, formatted_img_msg]
                for i, formatted_img_msg in enumerate(formatted_images)
            ]
            content.extend(sum(interleaved_content, []))
        else:
            # Text-only message
            content = [{"type": "text", "text": vlm_input.prompt}]
        return content

    def _prepare_historical_content(self, history: VLMHistory) -> List[Dict[str, Any]]:
        """Prepare historical content to be included in the content array.

        Args:
            history: The conversation history

        Returns:
            List of content items to be included in the message content
        """
        content = []

        # Add the history prompt if provided
        if history.prompt:
            content.append({"type": "text", "text": history.prompt})

        # Add each historical input's content
        for vlm_input in history.vlm_inputs:
            content.extend(self._prepare_content(vlm_input))

        return content

    def _estimate_prompt_tokens(self, prompt: str, n_images: int, model: str) -> int:
        """Roughly estimate the number of tokens in a prompt.

        Args:
            prompt: The text prompt
            n_images: Number of images in the prompt
            model: The model name

        Returns:
            Estimated number of tokens
        """
        # These are very rough estimates and should be replaced with actual tokenizers

        # Estimate text tokens (roughly 4 chars per token)
        text_tokens = len(prompt) // 4

        # TODO -- actually calculate w tiktoken? or handle images for 4o/4o-mini?
        # Estimate image tokens (models handle this differently)
        if model.startswith("gpt-4-vision"):
            # GPT-4V uses about 500 tokens per image on "low" detail level
            image_tokens = n_images * 500
        elif model.startswith("claude-3"):
            # Claude 3 uses about 500 tokens per image as a rough estimate
            image_tokens = n_images * 500
        else:
            # Generic fallback
            image_tokens = n_images * 500

        return text_tokens + image_tokens

    async def generate(self, request: VLMRequest) -> VLMResponse:
        """Generate a response from a VLM model.

        Args:
            request: The VLM request

        Returns:
            The VLM response

        Raises:
            Exception: If the request fails
        """
        # Determine provider if not specified
        provider = request.provider or get_provider_from_model(request.model)
        # Prepare messages
        messages = []

        # Add system prompt if provided
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        # Prepare user message with images if any
        user_message: Dict[str, Any] = {"role": "user"}

        # Prepare content for the user message
        vlm_input = request.vlm_input
        content = self._prepare_content(vlm_input)

        # Add history if provided
        if request.history:
            historical_content = self._prepare_historical_content(request.history)
            if request.history.placement == "before":
                content = historical_content + content
            elif request.history.placement == "after":
                content = content + historical_content
            else:
                raise ValueError(f"Invalid placement: {request.history.placement}")

        user_message["content"] = content
        messages.append(user_message)

        # Calculate estimated tokens
        estimated_tokens = self._estimate_prompt_tokens(
            request.vlm_input.prompt,
            len(request.vlm_input.images) if request.vlm_input.images else 0,
            request.model,
        )

        # Wait for rate limit
        await rate_limiter.wait_and_acquire(
            provider, request.model, estimated_tokens + request.max_tokens
        )

        start_time = time.time()
        try:
            # Make the API call
            response = await litellm.acompletion(
                model=request.model,
                messages=messages,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                timeout=request.timeout,
                stream=request.stream,
                **request.extra_params,
            )

            # Calculate response time
            response_ms = int((time.time() - start_time) * 1000)

            # Extract the text from the response
            if request.stream:
                # For streaming, we need to collect chunks
                chunks = []
                async for chunk in response:
                    chunks.append(chunk)

                # Combine chunks to get the full text
                text = "".join(
                    [
                        chunk.choices[0].delta.content
                        for chunk in chunks
                        if chunk.choices[0].delta.content is not None
                    ]
                )

                # Get usage from the last chunk
                usage = chunks[-1].usage.model_dump() if hasattr(chunks[-1], "usage") else {}
            else:
                # For non-streaming, just get the text directly
                text = response.choices[0].message.content
                usage = response.usage.model_dump() if hasattr(response, "usage") else {}

            # Update actual token usage
            if usage.get("total_tokens"):
                rate_limiter.record_usage(provider, request.model, usage["total_tokens"])

            # Return formatted response
            try:
                return VLMResponse(
                    text=text,
                    model=request.model,
                    provider=provider,
                    usage=usage,
                    response_ms=response_ms,
                    raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
                )
            except Exception as e:
                logger.error(
                    f"Error parsing response from {provider}/{request.model}: {str(e)}. Found response: {response.json()}"
                )
                raise

        except Exception as e:
            logger.error(
                f"Error generating response from {provider}/{request.model}: {str(e)}; {traceback.format_exc()}"
            )
            raise

    async def generate_parallel(
        self, requests: List[VLMRequest]
    ) -> List[Tuple[int, Optional[VLMResponse], Optional[str]]]:
        """Generate responses from multiple VLM requests in parallel.

        Args:
            requests: List of VLM requests

        Returns:
            List of tuples (index, response, exception) where:
                - index is the index of the request in the input list
                - response is the VLM response if successful, None otherwise
                - exception is the exception if failed, None otherwise
        """

        async def _process_request(
            index: int, request: VLMRequest
        ) -> Tuple[int, Optional[VLMResponse], Optional[str]]:
            try:
                response = await self.generate(request)
                return index, response, None
            except Exception:
                return index, None, traceback.format_exc()

        # Process all requests in parallel
        tasks = [_process_request(i, req) for i, req in enumerate(requests)]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return results


def load_image(image_path_or_bytes: t.Union[str, bytes]) -> bytes:
    """Load the image for a given task.

    Args:
        image_path_or_bytes: Either a file path to an image or the raw image bytes

    Returns:
        The image bytes
    """
    if isinstance(image_path_or_bytes, bytes):
        return image_path_or_bytes

    with open(image_path_or_bytes, "rb") as f:
        return f.read()


def create_vlm_request(
    model: str, image_path_or_bytes: t.Union[str, bytes], env_desc: str, task_desc: str
) -> VLMRequest:
    """Create a VLM request for the given configuration."""
    image_data = load_image(image_path_or_bytes)
    return VLMRequest(
        model=model,
        vlm_input=VLMInput(
            prompt=build_prompt(env_desc, task_desc),
            images=[
                ImageInput(data=base64.b64encode(image_data).decode("utf-8"), mime_type="image/png")
            ],
        ),
    )


def parse_vlm_response(response_text: str) -> t.Tuple[str, dict]:
    """Parse the VLM response into description and structured answer."""
    description = response_text.split("```json")[0]

    # Find the JSON content between ```json and ``` markers
    json_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", response_text)
    if not json_match:
        raise ValueError("Could not find JSON content between ``` markers")

    json_str = json_match.group(1)

    # Clean up common JSON formatting issues
    json_str = re.sub(r",(\s*})", r"\1", json_str)  # Remove trailing commas before closing braces

    try:
        return description.strip(), json.loads(json_str)
    except json.JSONDecodeError as e:
        # Add context about the actual JSON string that failed to parse
        raise ValueError(f"Failed to parse JSON: {str(e)}\nJSON string was:\n{json_str}") from e


def parse_vlm_responses(
    responses: List[Tuple[int, Optional[VLMResponse], Optional[str]]],
) -> List[Dict[str, Any]]:
    results = []
    for i, response, maybe_exception in responses:
        if maybe_exception or response is None:
            results.append({"index": i, "error": str(maybe_exception), "success": False})
        else:
            try:
                description, answer = parse_vlm_response(response.text)
                results.append(
                    {"index": i, "description": description, "answer": answer, "success": True}
                )
            except Exception as e:
                results.append(
                    {"index": i, "error": str(e), "raw_response": response.text, "success": False}
                )
    return results
