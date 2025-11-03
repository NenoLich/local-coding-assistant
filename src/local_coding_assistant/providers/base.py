"""
Base classes for LLM providers
"""

import abc
import asyncio
import os
from collections.abc import AsyncGenerator
from typing import Any

from pydantic import BaseModel, Field, field_validator

from local_coding_assistant.utils.logging import get_logger

logger = get_logger("providers.base")


class ProviderLLMRequest(BaseModel):
    """Standardized request format for all providers with validation"""

    messages: list[dict[str, Any]] = Field(
        description="List of message objects with role and content"
    )
    model: str = Field(
        description="Model identifier to use for generation", min_length=1
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for response generation",
    )
    max_tokens: int | None = Field(
        default=None, gt=0, description="Maximum number of tokens to generate"
    )
    stream: bool = Field(default=False, description="Whether to stream the response")
    tools: list[dict[str, Any]] | None = Field(
        default=None, description="Available tools/functions for the model"
    )
    tool_choice: str | dict[str, Any] | None = Field(
        default=None, description="Tool choice specification"
    )
    response_format: dict[str, Any] | None = Field(
        default=None, description="Response format specification"
    )
    tool_outputs: dict[str, Any] | None = Field(
        default=None, description="Tool outputs from the model"
    )
    extra_params: dict[str, Any] | None = Field(
        default=None, description="Additional provider-specific parameters"
    )

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v):
        """Ensure messages are properly formatted"""
        if not v:
            raise ValueError("messages cannot be empty")
        for msg in v:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise ValueError("Each message must have 'role' and 'content' fields")
        return v


class ProviderLLMResponse(BaseModel):
    """Standardized response format from all providers with validation"""

    content: str = Field(description="Generated text content")
    model: str = Field(description="Model that generated the response", min_length=1)
    tokens_used: int | None = Field(
        default=None, ge=0, description="Number of tokens used in the request"
    )
    finish_reason: str | None = Field(
        default=None, description="Reason why generation finished"
    )
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None, description="Tool calls made by the model"
    )
    usage: dict[str, Any] | None = Field(
        default=None, description="Detailed usage statistics"
    )
    metadata: dict[str, Any] | None = Field(
        default=None, description="Additional metadata about the response"
    )

    @field_validator("content")
    @classmethod
    def validate_content(cls, v):
        """Ensure content is a string"""
        if not isinstance(v, str):
            raise ValueError("content must be a string")
        return v


class ProviderLLMResponseDelta(BaseModel):
    """Standardized streaming response delta format"""

    content: str = Field(
        default="", description="Content chunk from streaming response"
    )
    role: str | None = Field(
        default=None, description="Role of the message (for streaming)"
    )
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None, description="Tool calls in this delta"
    )
    finish_reason: str | None = Field(
        default=None, description="Finish reason for this chunk"
    )
    metadata: dict[str, Any] | None = Field(
        default={}, description="Additional metadata for this chunk"
    )


class BaseDriver(abc.ABC):
    """Base class for LLM drivers that handle the actual API calls"""

    def __init__(self, api_key: str | None, base_url: str, **kwargs):
        self.api_key = api_key
        self.base_url = base_url
        self.kwargs = kwargs

    @abc.abstractmethod
    async def generate(self, request: ProviderLLMRequest) -> ProviderLLMResponse:
        """Generate a response from the LLM"""
        pass

    async def stream(
        self, request: ProviderLLMRequest
    ) -> AsyncGenerator[ProviderLLMResponseDelta, None]:
        """Generate a streaming response from the LLM"""
        # Default implementation - should be overridden by subclasses
        response = await self.generate(request)
        # Convert response to single delta (for non-streaming providers)
        yield ProviderLLMResponseDelta(
            content=response.content,
            finish_reason=response.finish_reason,
            metadata=response.metadata,
        )

    @abc.abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is healthy and accessible"""
        pass

    async def generate_with_retry(
        self,
        request: ProviderLLMRequest,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> ProviderLLMResponse:
        """Generate with retry logic"""
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                return await self.generate(request)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    await asyncio.sleep(
                        retry_delay * (2**attempt)
                    )  # Exponential backoff

        raise last_error or RuntimeError(f"Failed after {max_retries} attempts")

    async def stream_with_retry(
        self,
        request: ProviderLLMRequest,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> AsyncGenerator[ProviderLLMResponseDelta, None]:
        """Stream with retry logic"""
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                async for delta in self.stream(request):
                    yield delta
                return
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    await asyncio.sleep(
                        retry_delay * (2**attempt)
                    )  # Exponential backoff

        raise last_error or RuntimeError(f"Failed after {max_retries} attempts")


class BaseProvider(abc.ABC):
    """Base class for LLM providers

    Subclasses must implement the _create_driver_instance method to provide
    a driver instance for the specific provider implementation.
    """

    @abc.abstractmethod
    def _create_driver_instance(self) -> BaseDriver:
        """Create and return a driver instance for this provider.

        This is a required method that must be implemented by all provider subclasses.
        It should return an instance of a class that inherits from BaseDriver.

        Returns:
            An instance of a BaseDriver subclass specific to this provider
        """
        raise NotImplementedError("Subclasses must implement _create_driver_instance")

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str | None = None,
        api_key_env: str | None = None,
        models: list[str] | None = None,
        driver: str = "openai_chat",
        health_check_endpoint: str | None = None,
        health_check_method: str = "GET",
        health_check_timeout: float = 5.0,
        **kwargs,
    ):
        self.name = name
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.models = models or []
        self.driver = driver
        self.health_check_endpoint = health_check_endpoint
        self.health_check_method = health_check_method.upper()
        self.health_check_timeout = health_check_timeout
        self.kwargs = kwargs

        # Resolve API key
        self.api_key = api_key or self._get_api_key()

        if not self.api_key:
            from local_coding_assistant.providers.exceptions import ProviderError

            raise ProviderError(f"No API key provided for provider {name}")

        # Initialize driver instance immediately
        self.driver_instance = self._create_driver_instance()

    def _get_api_key(self) -> str | None:
        """Get API key from environment variable if specified"""
        if self.api_key_env:
            import os

            return os.getenv(self.api_key_env)
        return None

    async def _handle_auth_error(self, error: Exception) -> None:
        """Handle authentication errors and try to recover with API key from env var if available"""
        from local_coding_assistant.providers.exceptions import ProviderAuthError

        if not isinstance(error, ProviderAuthError):
            raise error

        # If we have an API key env var, and it's different from the current key, try with that
        if (
            self.api_key_env
            and (env_key := self._get_api_key())
            and env_key != self.api_key
        ):
            logger.info(
                f"Authentication failed, trying with key from {self.api_key_env}"
            )
            self.api_key = env_key
            # Recreate the driver instance with the new API key
            self.driver_instance = self._create_driver_instance()
            return

        # If we get here, we couldn't recover from the auth error
        raise error

    def _initialize_driver(self):
        """
        Initialize driver instance based on driver type.

        This is a helper method that can be used by subclasses to create
        standard driver instances. Subclasses can override this or implement
        _create_driver_instance directly.
        """
        driver_mapping = {
            "openai_chat": "OpenAIChatCompletionsDriver",
            "openai_responses": "OpenAIResponsesDriver",
            "local": "LocalDriver",
        }

        driver_name = driver_mapping.get(self.driver)
        if not driver_name:
            from local_coding_assistant.providers.exceptions import ProviderError

            raise ProviderError(f"Unknown driver type: {self.driver}")

        # Import the driver class locally to avoid circular imports
        from local_coding_assistant.providers.compatible_drivers import (
            LocalDriver,
            OpenAIChatCompletionsDriver,
            OpenAIResponsesDriver,
        )

        driver_classes = {
            "OpenAIChatCompletionsDriver": OpenAIChatCompletionsDriver,
            "OpenAIResponsesDriver": OpenAIResponsesDriver,
            "LocalDriver": LocalDriver,
        }

        driver_class = driver_classes.get(driver_name)
        if not driver_class:
            from local_coding_assistant.providers.exceptions import ProviderError

            raise ProviderError(f"Unknown driver type: {self.driver}")

        # Avoid passing provider_name twice if it's already in kwargs
        driver_kwargs = self.kwargs.copy()
        if "provider_name" not in driver_kwargs:
            driver_kwargs["provider_name"] = self.name

        return driver_class(
            api_key=self.api_key,
            base_url=self.base_url,
            **driver_kwargs,
        )

    async def generate(self, request: ProviderLLMRequest) -> ProviderLLMResponse:
        """Generate a response using this provider"""
        try:
            return await self.driver_instance.generate(request)
        except Exception as e:
            await self._handle_auth_error(e)
            # If we get here, _handle_auth_error didn't raise, so we can retry
            return await self.driver_instance.generate(request)

    async def generate_with_retry(
        self,
        request: ProviderLLMRequest,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> ProviderLLMResponse:
        """Generate a response with retry logic"""
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                return await self.generate(request)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    await asyncio.sleep(
                        retry_delay * (2**attempt)
                    )  # Exponential backoff

        raise last_error or RuntimeError(f"Failed after {max_retries} attempts")

    async def stream(
        self, request: ProviderLLMRequest
    ) -> AsyncGenerator[ProviderLLMResponseDelta, None]:
        """Generate a streaming response using this provider"""
        try:
            if self.driver_instance:
                async for delta in self.driver_instance.stream(request):
                    yield delta
            else:
                # Fallback to non-streaming response as single delta
                response = await self.generate(request)
                yield ProviderLLMResponseDelta(
                    content=response.content,
                    finish_reason=response.finish_reason,
                    metadata=response.metadata,
                )
        except Exception as e:
            await self._handle_auth_error(e)
            # If we get here, _handle_auth_error didn't raise, so we can retry
            if self.driver_instance:
                async for delta in self.driver_instance.stream(request):
                    yield delta
            else:
                # Fallback to non-streaming response as single delta
                response = await self.generate(request)
                yield ProviderLLMResponseDelta(
                    content=response.content,
                    finish_reason=response.finish_reason,
                    metadata=response.metadata,
                )

    async def stream_with_retry(
        self,
        request: ProviderLLMRequest,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> AsyncGenerator[ProviderLLMResponseDelta, None]:
        """Stream with retry logic"""
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                async for delta in self.stream(request):
                    yield delta
                return
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    await asyncio.sleep(
                        retry_delay * (2**attempt)
                    )  # Exponential backoff

        raise last_error or RuntimeError(f"Failed after {max_retries} attempts")

    def _get_health_check_url(self) -> str:
        """Get the full URL for the health check endpoint."""
        if not self.health_check_endpoint:
            return self.base_url.rstrip("/")

        if self.health_check_endpoint.startswith(("http://", "https://")):
            return self.health_check_endpoint

        return f"{self.base_url.rstrip('/')}/{self.health_check_endpoint.lstrip('/')}"

    async def _make_health_request(self, url: str, api_key: str) -> bool:
        """Make the actual health check request."""
        import httpx

        from local_coding_assistant.providers.exceptions import (
            ProviderAuthError,
            ProviderError,
        )

        try:
            async with httpx.AsyncClient(timeout=self.health_check_timeout) as client:
                response = await client.request(
                    method=self.health_check_method,
                    url=url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=self.health_check_timeout,
                )
                response.raise_for_status()
                return True

        except httpx.HTTPStatusError as err:
            if err.response.status_code == 401:
                raise ProviderAuthError(
                    f"Authentication failed for {self.name}"
                ) from err
            logger.warning(
                f"Health check failed for {self.name} with status {err.response.status_code}"
            )
            return False

        except Exception as err:
            error_msg = f"Health check error for {self.name}: {err!s}"
            logger.error(error_msg)
            raise ProviderError(error_msg) from err

    async def _try_env_api_key(self, url: str, original_error: Exception) -> bool:
        """Try using API key from environment variable if available."""
        if not (
            self.api_key_env
            and (env_key := os.getenv(self.api_key_env))
            and env_key != self.api_key
        ):
            raise original_error

        logger.info(
            f"Authentication failed with current API key, trying with key from {self.api_key_env}"
        )
        from local_coding_assistant.providers import ProviderAuthError

        try:
            result = await self._make_health_request(url, env_key)
            self._update_api_key(env_key)
            return result
        except ProviderAuthError:
            raise original_error from None

    def _update_api_key(self, new_key: str) -> None:
        """Update the API key and recreate the driver instance."""
        self.api_key = new_key
        self.driver_instance = self._create_driver_instance()
        logger.info(
            f"Successfully updated API key for {self.name} from environment variable"
        )

    async def health_check(self) -> bool | str:
        """Check provider health.

        Returns:
            bool: True if health check passed, False if failed
            str: 'unavailable' if health check is not implemented or not configured

        Raises:
            ProviderError: If there's an error during health check
        """
        if not self.health_check_endpoint:
            return "unavailable"

        from local_coding_assistant.providers.exceptions import ProviderAuthError

        url = self._get_health_check_url()

        try:
            if self.api_key is None:
                raise ProviderAuthError("No API key provided")
            return await self._make_health_request(url, self.api_key)

        except ProviderAuthError as e:
            return await self._try_env_api_key(url, e)

    def supports_model(self, model: str) -> bool:
        """Check if this provider supports the given model"""
        return model in self.models

    def get_available_models(self) -> list[str]:
        """Get list of available models for this provider"""
        return self.models.copy()
