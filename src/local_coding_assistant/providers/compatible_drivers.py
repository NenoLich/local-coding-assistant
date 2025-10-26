"""
Compatible drivers for different LLM APIs

This module provides standardized drivers for common LLM APIs like OpenAI,
ensuring consistent behavior across different providers.
"""

from collections.abc import AsyncGenerator
from typing import NoReturn

from openai import AsyncOpenAI

from local_coding_assistant.providers.base import (
    BaseDriver,
    ProviderLLMRequest,
    ProviderLLMResponse,
    ProviderLLMResponseDelta,
)
from local_coding_assistant.providers.exceptions import (
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
)
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("providers.compatible_drivers")


class OpenAIChatCompletionsDriver(BaseDriver):
    """Driver for OpenAI-compatible chat.completions API using the openai library"""

    def __init__(self, api_key: str | None, base_url: str, **kwargs):
        super().__init__(api_key, base_url, **kwargs)
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def generate(self, request: ProviderLLMRequest) -> ProviderLLMResponse:
        """Generate using OpenAI chat.completions API"""
        payload = {
            "model": request.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "stream": request.stream,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.tools:
            payload["tools"] = request.tools
            payload["tool_choice"] = request.tool_choice
        if request.response_format:
            payload["response_format"] = request.response_format
        if request.extra_params:
            payload.update(request.extra_params)

        try:
            response = await self.client.chat.completions.create(**payload)
            return self._parse_response(response, request.model)
        except Exception as e:
            self._handle_error(e)

    async def stream(
        self, request: ProviderLLMRequest
    ) -> AsyncGenerator[ProviderLLMResponseDelta, None]:
        """Generate a streaming response using OpenAI chat.completions API"""
        payload = {
            "model": request.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "stream": True,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.tools:
            payload["tools"] = request.tools
            payload["tool_choice"] = request.tool_choice
        if request.extra_params:
            payload.update(request.extra_params)

        try:
            stream = await self.client.chat.completions.create(**payload)
            async for chunk in stream:
                choice = chunk.choices[0]
                delta = choice.delta
                yield ProviderLLMResponseDelta(
                    content=delta.content or "",
                    role=delta.role,
                    tool_calls=delta.tool_calls,
                    finish_reason=choice.finish_reason,
                    metadata={
                        "response_id": chunk.id,
                        "created": chunk.created,
                        "model": chunk.model,
                    },
                )
        except Exception as e:
            self._handle_error(e)

    def _parse_response(self, response, model: str) -> ProviderLLMResponse:
        """Parse OpenAI API response"""
        choice = response.choices[0]
        message = choice.message
        return ProviderLLMResponse(
            content=message.content or "",
            model=model,
            tokens_used=self._calculate_tokens(response),
            finish_reason=choice.finish_reason,
            tool_calls=message.tool_calls,
            usage=dict(response.usage) if hasattr(response, "usage") else None,
            metadata={
                "response_id": response.id,
                "created": response.created,
                "model": response.model,
            },
        )

    def _calculate_tokens(self, response) -> int | None:
        """Calculate total tokens used"""
        return getattr(response.usage, "total_tokens", None)

    async def health_check(self) -> bool:
        """Check if the API is accessible"""
        try:
            await self.client.models.list()
            return True
        except Exception:
            return False

    def _handle_error(self, e: Exception) -> NoReturn:
        """Handle API errors"""
        if "401" in str(e):
            raise ProviderAuthError(
                "Invalid API key", provider=self.kwargs.get("provider_name")
            )
        elif "429" in str(e):
            raise ProviderRateLimitError(
                "Rate limit exceeded", provider=self.kwargs.get("provider_name")
            )
        elif "500" in str(e):
            raise ProviderConnectionError(
                "Server error", provider=self.kwargs.get("provider_name")
            )
        else:
            logger.error(
                f"API error in {self.kwargs.get('provider_name', 'unknown')}: {e!s}"
            )
            raise ProviderError(
                f"API error: {e!s}", provider=self.kwargs.get("provider_name")
            )


class OpenAIResponsesDriver(BaseDriver):
    """Driver for OpenAI responses API using the openai library"""

    def __init__(self, api_key: str | None, base_url: str, **kwargs):
        super().__init__(api_key, base_url, **kwargs)
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def generate(self, request: ProviderLLMRequest) -> ProviderLLMResponse:
        """Generate using OpenAI responses API"""
        payload = {
            "model": request.model,
            "input": request.messages,
            "temperature": request.temperature,
        }
        if request.max_tokens:
            payload["max_output_tokens"] = request.max_tokens
        if request.tools:
            payload["tools"] = request.tools
        if request.extra_params:
            payload.update(request.extra_params)

        try:
            response = await self.client.responses.create(**payload)
            return self._parse_response(response, request.model)
        except Exception as e:
            self._handle_error(e)

    async def stream(
        self, request: ProviderLLMRequest
    ) -> AsyncGenerator[ProviderLLMResponseDelta, None]:
        """Generate a streaming response using OpenAI responses API"""
        payload = {
            "model": request.model,
            "input": request.messages,
            "temperature": request.temperature,
            "stream": True,
        }
        if request.max_tokens:
            payload["max_output_tokens"] = request.max_tokens
        if request.tools:
            payload["tools"] = request.tools
        if request.extra_params:
            payload.update(request.extra_params)

        try:
            stream = await self.client.responses.create(**payload)
            async for chunk in stream:
                yield ProviderLLMResponseDelta(
                    content=chunk.output_text or "",
                    finish_reason=chunk.finish_reason,
                    metadata={
                        "response_id": getattr(chunk, "id", None),
                        "created": getattr(chunk, "created", None),
                        "model": getattr(chunk, "model", None),
                    },
                )
        except Exception as e:
            self._handle_error(e)

    def _parse_response(self, response, model: str) -> ProviderLLMResponse:
        """Parse OpenAI responses API response"""
        return ProviderLLMResponse(
            content=response.output_text,
            model=model,
            tokens_used=getattr(response, "usage", {}).get("total_tokens"),
            finish_reason=getattr(response, "finish_reason", None),
            usage=getattr(response, "usage", None),
            metadata={
                "response_id": getattr(response, "id", None),
                "created": getattr(response, "created", None),
                "model": getattr(response, "model", None),
            },
        )

    async def health_check(self) -> bool:
        """Check if the API is accessible"""
        try:
            await self.client.models.list()
            return True
        except Exception:
            return False

    def _handle_error(self, e: Exception) -> NoReturn:
        """Handle API errors"""
        if "401" in str(e):
            raise ProviderAuthError(
                "Invalid API key", provider=self.kwargs.get("provider_name")
            )
        elif "429" in str(e):
            raise ProviderRateLimitError(
                "Rate limit exceeded", provider=self.kwargs.get("provider_name")
            )
        elif "500" in str(e):
            raise ProviderConnectionError(
                "Server error", provider=self.kwargs.get("provider_name")
            )
        else:
            logger.error(
                f"API error in {self.kwargs.get('provider_name', 'unknown')}: {e!s}"
            )
            raise ProviderError(
                f"API error: {e!s}", provider=self.kwargs.get("provider_name")
            )


class LocalDriver(BaseDriver):
    """Driver for local/offline models (placeholder for future implementation)"""

    def __init__(self, api_key: str | None, base_url: str, **kwargs):
        super().__init__(api_key, base_url, **kwargs)

    async def generate(self, request: ProviderLLMRequest) -> ProviderLLMResponse:
        """Generate using local model (placeholder)"""
        # This would integrate with local LLM servers like Ollama, vLLM, etc.
        raise NotImplementedError("Local driver not yet implemented")

    async def health_check(self) -> bool:
        """Check if local model is available"""
        return False
