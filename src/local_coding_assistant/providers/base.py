"""
Base classes for LLM providers
"""

import abc
import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from pydantic import BaseModel, Field, field_validator

from local_coding_assistant.providers.exceptions import ProviderError


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
    """Base class for LLM providers"""

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str | None = None,
        api_key_env: str | None = None,
        models: list[str] | None = None,
        driver: str = "openai_chat",
        **kwargs,
    ):
        self.name = name
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.models = models or []
        self.driver = driver
        self.kwargs = kwargs

        # Resolve API key
        self.api_key = api_key or self._get_api_key()

        if not self.api_key:
            raise ProviderError(f"No API key provided for provider {name}")

        # Initialize driver instance immediately
        self.driver_instance = self._create_driver_instance()

    def _get_api_key(self) -> str | None:
        """Get API key from environment variable if specified"""
        if self.api_key_env:
            import os

            return os.getenv(self.api_key_env)
        return None

    def _create_driver_instance(self) -> BaseDriver:
        """Create and return the appropriate driver instance"""
        # If a specific driver is requested, use that
        if self.driver:
            return self._initialize_driver()

        # Use default OpenAI-compatible driver
        from local_coding_assistant.providers.compatible_drivers import (
            OpenAIChatCompletionsDriver,
        )

        # Avoid passing provider_name twice if it's already in kwargs
        driver_kwargs = self.kwargs.copy()
        if "provider_name" not in driver_kwargs:
            driver_kwargs["provider_name"] = self.name

        return OpenAIChatCompletionsDriver(
            api_key=self.api_key,
            base_url=self.base_url,
            **driver_kwargs,
        )

    def _initialize_driver(self):
        """Initialize driver instance based on driver type (for backwards compatibility)"""
        driver_mapping = {
            "openai_chat": "OpenAIChatCompletionsDriver",
            "openai_responses": "OpenAIResponsesDriver",
            "local": "LocalDriver",
        }

        driver_name = driver_mapping.get(self.driver)
        if not driver_name:
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

    @abc.abstractmethod
    async def generate(self, request: ProviderLLMRequest) -> ProviderLLMResponse:
        """Generate a response using this provider"""
        pass

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

    async def health_check(self) -> bool:
        """Check provider health"""
        if self.driver_instance:
            return await self.driver_instance.health_check()
        return False

    def supports_model(self, model: str) -> bool:
        """Check if this provider supports the given model"""
        return model in self.models

    def get_available_models(self) -> list[str]:
        """Get list of available models for this provider"""
        return self.models.copy()
