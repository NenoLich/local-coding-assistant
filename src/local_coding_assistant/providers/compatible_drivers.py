"""
Compatible drivers for different LLM APIs

This module provides standardized drivers for common LLM APIs like OpenAI,
ensuring consistent behavior across different providers.
"""

import json
from collections.abc import AsyncGenerator
from typing import Any, NoReturn

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
        # Start with required parameters
        payload = {
            "model": request.model,
            "messages": request.messages,
            "temperature": request.temperature,
        }

        # Add all parameters from request.parameters
        if request.parameters:
            # Get all set parameters, excluding None values
            params = {
                k: v
                for k, v in request.parameters.model_dump(exclude_unset=True).items()
                if v is not None
            }
            # Ensure stream is False for non-streaming
            params["stream"] = False
            payload.update(params)

        try:
            response = await self.client.chat.completions.create(**payload)
            logger.debug("Response: %s", response)
            return self._parse_response(response, request.model)
        except Exception as e:
            logger.error("Error in OpenAI API request", exc_info=True)
            self._handle_error(e)

    async def stream(
        self, request: ProviderLLMRequest
    ) -> AsyncGenerator[ProviderLLMResponseDelta, None]:
        """Generate a streaming response using OpenAI chat.completions API"""
        # Start with required parameters
        payload = {
            "model": request.model,
            "messages": request.messages,
            "temperature": request.temperature,
        }

        # Add all parameters from request.parameters
        if request.parameters:
            # Get all set parameters, excluding None values
            params = {
                k: v
                for k, v in request.parameters.model_dump(exclude_unset=True).items()
                if v is not None
            }
            # Ensure stream is True for streaming
            params["stream"] = True
            payload.update(params)

        try:
            stream = await self.client.chat.completions.create(**payload)
            async for chunk in stream:
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                yield ProviderLLMResponseDelta(
                    content=delta.content or "",
                    role=getattr(delta, "role", None),
                    tool_calls=getattr(delta, "tool_calls", None),
                    finish_reason=choice.finish_reason,
                    metadata={
                        "response_id": getattr(chunk, "id", None),
                        "created": getattr(chunk, "created", None),
                        "model": getattr(chunk, "model", request.model),
                    },
                )
        except Exception as e:
            logger.error("Error in OpenAI streaming request", exc_info=True)
            self._handle_error(e)

    def _parse_response(self, response, model: str) -> ProviderLLMResponse:
        """Parse OpenAI API response, handling both standard and GitHub Models API formats."""
        try:
            choice = response.choices[0]
            message = choice.message

            # Parse tool calls if present
            tool_calls = []
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call_entry in message.tool_calls:
                    parsed_call = self._parse_tool_call(tool_call_entry)
                    if parsed_call:
                        tool_calls.append(parsed_call)

            return ProviderLLMResponse(
                content=message.content or "",
                model=model,
                tokens_used=self._calculate_tokens(response),
                finish_reason=choice.finish_reason,
                tool_calls=tool_calls or None,
                usage=dict(response.usage) if hasattr(response, "usage") else None,
                metadata={
                    "response_id": getattr(response, "id", None),
                    "created": getattr(response, "created", None),
                    "model": getattr(response, "model", model),
                },
            )
        except Exception as e:
            logger.error(f"Error parsing response: {e}", exc_info=True)
            raise ProviderError(f"Failed to parse response: {e}") from e

    def _parse_tool_call(self, tool_call: Any) -> dict[str, Any]:
        """Helper to parse a single tool call into a standardized format."""
        try:
            # Convert tool_call to dict (Pydantic v1/v2 or raw dict)
            if hasattr(tool_call, "model_dump"):
                tc = tool_call.model_dump()
            elif hasattr(tool_call, "dict"):
                tc = tool_call.dict()
            elif isinstance(tool_call, dict):
                tc = tool_call
            else:
                tc = {
                    "id": getattr(tool_call, "id", None),
                    "type": getattr(tool_call, "type", "function"),
                    "function": {
                        "name": getattr(
                            getattr(tool_call, "function", None), "name", ""
                        ),
                        "arguments": getattr(
                            getattr(tool_call, "function", None), "arguments", "{}"
                        ),
                    },
                }

            # Ensure tc is a dictionary
            if not isinstance(tc, dict):
                tc = {}

            # Get function data, defaulting to an empty dict
            function_data = tc.get("function")

            # Ensure function_data is a dictionary
            if not isinstance(function_data, dict):
                function_data = {}

            # Create a new dictionary to ensure type safety
            function_data = dict(function_data)

            # Convert arguments to JSON string if it's a dict
            arguments = function_data.get("arguments")
            if isinstance(arguments, dict):
                function_data["arguments"] = json.dumps(arguments)

            # Build the result
            return {
                "id": str(tc.get("id", f"call_{id(tool_call)}")),
                "type": str(tc.get("type", "function")),
                "function": {
                    "name": str(function_data.get("name", "")),
                    "arguments": str(function_data.get("arguments", "{}")),
                },
            }
        except Exception as err:
            logger.warning(f"Failed to parse tool call: {err}", exc_info=True)
            return {
                "id": f"error_{id(err)}",
                "type": "function",
                "function": {
                    "name": "error",
                    "arguments": f'{{ "error": "{str(err).replace("'", "\\'")}" }}',
                },
            }

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

    def _extract_error_details(self, e: Exception) -> tuple[int | None, Any]:
        """Extract status code and response from exception.

        Args:
            e: The exception to extract details from

        Returns:
            Tuple of (status_code, response) where either or both can be None
        """
        if hasattr(e, "status_code"):
            status_code = int(e.status_code) if e.status_code is not None else None
            return status_code, getattr(e, "response", None)

        if hasattr(e, "response") and hasattr(e.response, "status_code"):
            status_code = (
                int(e.response.status_code)
                if e.response.status_code is not None
                else None
            )
            return status_code, e.response

        return None, None

    def _log_error_details(
        self, e: Exception, status_code: int | None, response: Any
    ) -> None:
        """Log details about the error for debugging purposes."""
        logger.debug(f"Error type: {type(e).__module__}.{type(e).__name__}")
        if status_code is not None:
            logger.debug(f"Detected status code: {status_code}")
        if response and hasattr(response, "text"):
            logger.debug(f"Response text: {response.text!r}")

    def _handle_http_error(self, status_code: int, error_message: str) -> NoReturn:
        """Handle HTTP errors with specific status codes."""
        provider = self.kwargs.get("provider_name")

        if status_code == 401:
            raise ProviderAuthError(
                f"Invalid API key (HTTP {status_code})",
                provider=provider,
            )
        if status_code == 429:
            raise ProviderRateLimitError(
                f"Rate limit exceeded (HTTP {status_code})",
                provider=provider,
            )
        if 500 <= status_code < 600:
            raise ProviderConnectionError(
                f"Server error (HTTP {status_code})",
                provider=provider,
            )

        error_msg = f"HTTP {status_code} error: {error_message}"
        logger.error(f"API error in {provider or 'unknown'}: {error_msg}")
        raise ProviderError(error_msg, provider=provider)

    def _handle_error(self, e: Exception) -> NoReturn:
        """Handle API errors by checking status code from error attributes"""
        status_code, response = self._extract_error_details(e)
        self._log_error_details(e, status_code, response)

        # Handle based on status code if available
        if status_code is not None:
            self._handle_http_error(status_code, str(e))

        # String-based error matching as fallback
        error_str = str(e)
        if "401" in error_str:
            raise ProviderAuthError(
                f"Authentication failed: {error_str}",
                provider=self.kwargs.get("provider_name"),
            )
        if "429" in error_str:
            raise ProviderRateLimitError(
                f"Rate limit exceeded: {error_str}",
                provider=self.kwargs.get("provider_name"),
            )

        # Default error handling
        provider_name = self.kwargs.get("provider_name", "unknown")
        logger.error(
            f"API error in {provider_name}: {error_str}",
            exc_info=True,
        )
        raise ProviderError(f"API error: {error_str}", provider=provider_name)


class OpenAIResponsesDriver(BaseDriver):
    """Driver for OpenAI responses API using the openai library"""

    def __init__(self, api_key: str | None, base_url: str, **kwargs):
        # Ensure base_url doesn't end with /responses
        base_url = base_url.rstrip("/")
        base_url = base_url.removesuffix("/responses")

        super().__init__(api_key, base_url, **kwargs)
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    def _format_tools_for_responses_api(self, tools: Any) -> list[dict]:
        """Format tools to be compatible with Responses API

        Args:
            tools: Input tools which can be a list of dicts, a single dict, or other types

        Returns:
            List of formatted tool dictionaries

        Raises:
            TypeError: If tools is not a list or dict
        """
        if tools is None:
            return []

        if not isinstance(tools, list | dict):
            raise TypeError(
                f"Expected list or dict for tools, got {type(tools).__name__}"
            )

        # Handle single tool as dict
        if isinstance(tools, dict):
            tools = [tools]

        formatted_tools = []
        for tool in tools:
            if "function" in tool:
                # Convert from Chat Completions format to Responses API format
                formatted_tool = {
                    "type": "function",
                    "name": tool["function"].get("name", ""),
                    "description": tool["function"].get("description", ""),
                    "parameters": tool["function"].get("parameters", {}),
                }
                formatted_tools.append(formatted_tool)
            elif "name" in tool and "parameters" in tool:
                # Tool is already in the expected format
                formatted_tools.append(tool)
            else:
                # Unsupported format, log a warning and include as-is
                logger.warning(f"Unsupported tool format: {tool}")
                formatted_tools.append(tool)

        return formatted_tools

    async def generate(self, request: ProviderLLMRequest) -> ProviderLLMResponse:
        """Generate using OpenAI responses API"""
        # Get the flattened request data
        request_data = request.model_dump()

        # Build the payload with all available parameters
        payload: dict[str, Any] = {
            "model": request.model,
            "input": request.messages,
            "temperature": request.temperature,
            **{
                k: v
                for k, v in request_data.items()
                if k not in ["model", "messages", "temperature", "parameters"]
                and v is not None
            },
        }

        # Handle max_tokens -> max_output_tokens mapping
        if "max_tokens" in payload:
            payload["max_output_tokens"] = payload.pop("max_tokens")

        # Format tools if present
        if "tools" in payload:
            payload["tools"] = self._format_tools_for_responses_api(payload["tools"])
            if "tool_choice" not in payload:
                payload["tool_choice"] = "auto"

        try:
            response = await self.client.responses.create(**payload)
            return self._parse_response(response, request.model)
        except Exception as e:
            self._handle_error(e)

    async def stream(
        self, request: ProviderLLMRequest
    ) -> AsyncGenerator[ProviderLLMResponseDelta, None]:
        """Generate a streaming response using OpenAI responses API"""
        # Get the flattened request data
        request_data = request.model_dump()

        # Build the payload with all available parameters
        payload: dict[str, Any] = {
            "model": request.model,
            "input": request.messages,
            "temperature": request.temperature,
            **{
                k: v
                for k, v in request_data.items()
                if k not in ["model", "messages", "temperature", "parameters"]
                and v is not None
            },
        }

        # Handle max_tokens -> max_output_tokens mapping
        if "max_tokens" in payload:
            payload["max_output_tokens"] = payload.pop("max_tokens")

        # Format tools if present
        if "tools" in payload:
            payload["tools"] = self._format_tools_for_responses_api(payload["tools"])
            if "tool_choice" not in payload:
                payload["tool_choice"] = "auto"

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
        # Extract usage data from ResponseUsage object
        usage = getattr(response, "usage", None)
        if usage is not None:
            # Convert ResponseUsage to dict if it's not already
            if hasattr(usage, "model_dump"):
                usage = usage.model_dump()
            elif not isinstance(usage, dict):
                usage = {
                    "total_tokens": getattr(usage, "total_tokens", None),
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                }

        # Parse tool calls if present in the response
        tool_calls = (
            self._parse_tool_calls(response) if hasattr(response, "output") else None
        )

        parsed_response = ProviderLLMResponse(
            content=response.output_text,
            model=model,
            tokens_used=usage.get("total_tokens") if isinstance(usage, dict) else None,
            finish_reason=getattr(response, "finish_reason", None),
            tool_calls=tool_calls,
            usage=usage,
            metadata={
                "response_id": getattr(response, "id", None),
                "created": getattr(response, "created", None),
                "model": getattr(response, "model", None),
            },
        )
        logger.debug(f"Response: {parsed_response}")
        return parsed_response

    def _parse_tool_calls(self, response) -> list[dict[str, Any]]:
        """Parse tool calls from the response output.

        Args:
            response: The response object from the API

        Returns:
            List of tool call objects in the standard format
        """
        tool_calls = []

        # Check if the response has output and it's a list
        if not hasattr(response, "output") or not isinstance(response.output, list):
            return []

        for output_item in response.output:
            # Check if this is a function call
            if (
                not isinstance(output_item, dict)
                or output_item.get("type") != "function_call"
            ):
                continue

            # Parse the function call
            function = output_item.get("function", {})
            if not function:
                continue

            try:
                # Parse the arguments if it's a string
                arguments = function.get("arguments", "{}")
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)

                tool_call = {
                    "id": output_item.get("id", f"call_{id(output_item)}"),
                    "type": "function",
                    "function": {
                        "name": function.get("name", ""),
                        "arguments": arguments,
                    },
                }
                tool_calls.append(tool_call)
            except (json.JSONDecodeError, AttributeError) as e:
                logger.warning(f"Failed to parse tool call arguments: {e}")
                continue

        return tool_calls

    async def health_check(self) -> bool:
        """Check if the API is accessible"""
        try:
            await self.client.models.list()
            return True
        except Exception:
            return False

    def _extract_error_details(self, e: Exception) -> tuple[int | None, Any]:
        """Extract status code and response from exception.

        Args:
            e: The exception to extract details from

        Returns:
            Tuple of (status_code, response) where either or both can be None
        """
        if hasattr(e, "status_code"):
            status_code = int(e.status_code) if e.status_code is not None else None
            return status_code, getattr(e, "response", None)
        if hasattr(e, "response") and hasattr(e.response, "status_code"):
            status_code = (
                int(e.response.status_code)
                if e.response.status_code is not None
                else None
            )
            return status_code, e.response
        return None, None

    def _log_error_details(
        self, e: Exception, status_code: int | None, response: Any
    ) -> None:
        """Log details about the error for debugging purposes."""
        logger.debug(f"Error type: {type(e).__module__}.{type(e).__name__}")
        if status_code is not None:
            logger.debug(f"Detected status code: {status_code}")
        if response and hasattr(response, "text"):
            logger.debug(f"Response text: {response.text!r}")

    def _handle_http_error(self, status_code: int, error_message: str) -> NoReturn:
        """Handle HTTP errors with specific status codes."""
        provider = self.kwargs.get("provider_name")

        if status_code == 401:
            raise ProviderAuthError(
                f"Invalid API key (HTTP {status_code})",
                provider=provider,
            )
        if status_code == 429:
            raise ProviderRateLimitError(
                f"Rate limit exceeded (HTTP {status_code})",
                provider=provider,
            )
        if 500 <= status_code < 600:
            raise ProviderConnectionError(
                f"Server error (HTTP {status_code})",
                provider=provider,
            )

        error_msg = f"HTTP {status_code} error: {error_message}"
        logger.error(f"API error in {provider or 'unknown'}: {error_msg}")
        raise ProviderError(error_msg, provider=provider)

    def _handle_error(self, e: Exception) -> NoReturn:
        """Handle API errors by checking status code from error attributes"""
        status_code, response = self._extract_error_details(e)
        self._log_error_details(e, status_code, response)

        # Handle based on status code if available
        if status_code is not None:
            self._handle_http_error(status_code, str(e))

        # String-based error matching as fallback
        error_str = str(e)
        if "401" in error_str:
            raise ProviderAuthError(
                f"Authentication failed: {error_str}",
                provider=self.kwargs.get("provider_name"),
            )
        if "429" in error_str:
            raise ProviderRateLimitError(
                f"Rate limit exceeded: {error_str}",
                provider=self.kwargs.get("provider_name"),
            )

        # Default error handling
        provider_name = self.kwargs.get("provider_name", "unknown")
        logger.error(
            f"API error in {provider_name}: {error_str}",
            exc_info=True,
        )
        raise ProviderError(f"API error: {error_str}", provider=provider_name)


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
