"""
Compatible drivers for different LLM APIs

This module provides standardized drivers for common LLM APIs like OpenAI,
ensuring consistent behavior across different providers.
"""

import json
import time
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from typing import Any, NoReturn

import litellm
from litellm import acompletion, get_valid_models

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


class ResponseNormalizer:
    """Normalizer for OpenAI Responses API streaming events"""

    def __init__(self, start_time: float):
        self.reasoning_accumulator = ""
        self.start_time = start_time

    def _handle_reasoning_text_delta(self, event) -> ProviderLLMResponseDelta:
        """Handle reasoning text delta event."""
        return ProviderLLMResponseDelta(
            reasoning=event.delta,
            metadata={
                "sequence_number": event.sequence_number,
                "item_id": event.item_id,
                "output_index": event.output_index,
                "content_index": event.content_index,
            },
        )

    def _handle_output_text_delta(self, event) -> ProviderLLMResponseDelta:
        """Handle output text delta event."""
        return ProviderLLMResponseDelta(
            content=event.delta,
            metadata={
                "sequence_number": event.sequence_number,
                "item_id": event.item_id,
                "output_index": event.output_index,
                "content_index": event.content_index,
            },
        )

    def _extract_tool_calls_from_response(self, response) -> list[dict[str, Any]]:
        """Extract tool calls from the response output with provider-specific fields."""
        tool_calls = []
        if not response.output:
            return tool_calls

        for output_item in response.output:
            if getattr(output_item, "type", None) == "function_call":
                call_id = getattr(output_item, "call_id", None) or getattr(
                    output_item, "id", None
                )
                name = getattr(output_item, "name", None)
                arguments = getattr(output_item, "arguments", None)
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                extra_content = getattr(output_item, "extra_content", None)

                tool_call = {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    },
                }

                # Add provider-specific fields if present
                if extra_content:
                    tool_call["extra_content"] = extra_content

                tool_calls.append(tool_call)
        return tool_calls

    def _handle_response_completed(self, event, response) -> ProviderLLMResponseDelta:
        """Handle response completed event."""
        tool_calls = self._extract_tool_calls_from_response(response)
        return ProviderLLMResponseDelta(
            tool_calls=tool_calls if tool_calls else None,
            finish_reason="completed",
            metadata={
                "sequence_number": event.sequence_number,
                "response_id": response.id,
                "latency_ms": time.perf_counter() - self.start_time,
                "usage": response.usage.model_dump()
                if hasattr(response.usage, "model_dump")
                else response.usage,
            },
        )

    async def normalize_stream(self, stream):
        """Normalize the async stream of events into ProviderLLMResponseDelta"""
        async for event in stream:
            delta = None
            if event.type == "response.reasoning_text.delta":
                delta = self._handle_reasoning_text_delta(event)
            elif event.type == "response.output_text.delta":
                delta = self._handle_output_text_delta(event)
            elif event.type == "response.function_call.delta":
                # Handle tool call deltas (arguments may be partial, accumulate in executor if needed)
                # Note: Don't yield tool_calls here, only in completed response
                pass
            elif event.type == "response.completed":
                response = event.response
                logger.debug("Completed response event", event_received=event)
                delta = self._handle_response_completed(event, response)
            # Other event types (done events) are ignored

            if delta is not None:
                yield delta


class OpenAIChatCompletionsDriver(BaseDriver):
    """Driver for OpenAI-compatible chat.completions API using litellm"""

    def __init__(self, api_key: str | None, base_url: str, **kwargs):
        super().__init__(api_key, base_url, **kwargs)
        self.provider_name = kwargs.get("provider_name")
        # litellm handles provider selection via model prefix

    def _build_chat_payload(  # noqa C901
        self, request: ProviderLLMRequest, stream: bool = False
    ) -> dict[str, Any]:
        """Build payload for chat completions API using litellm format."""
        # Use "openai/" prefix to tell litellm to use chat completion API
        # Provider is determined by base_url and api_key, not by model prefix
        model = f"openai/{request.model}"

        payload: dict[str, Any] = {
            "model": model,
            "messages": self._format_messages(request.messages),
            "temperature": request.temperature,
        }
        if stream:
            payload["stream"] = True

        # Add base_url and api_key for litellm
        payload["api_base"] = self.base_url
        payload["api_key"] = self.api_key

        # Handle provider-specific parameters via extra_body
        extra_body = {}
        if request.parameters:
            # Check for extra_body.thinking_config for Gemini
            if (
                hasattr(request.parameters, "extra_body")
                and request.parameters.extra_body
            ):
                extra_body.update(request.parameters.extra_body)

        if request.parameters:
            if request.parameters.max_tokens is not None:
                payload["max_tokens"] = request.parameters.max_tokens
            if request.parameters.top_p is not None:
                payload["top_p"] = request.parameters.top_p
            if request.parameters.top_k is not None:
                payload["top_k"] = request.parameters.top_k
            if request.parameters.frequency_penalty is not None:
                payload["frequency_penalty"] = request.parameters.frequency_penalty
            if request.parameters.presence_penalty is not None:
                payload["presence_penalty"] = request.parameters.presence_penalty
            if request.parameters.seed is not None:
                payload["seed"] = request.parameters.seed
            if request.parameters.stop is not None:
                payload["stop"] = request.parameters.stop
            if request.parameters.response_format is not None:
                payload["response_format"] = request.parameters.response_format
            if request.parameters.tools is not None:
                payload["tools"] = request.parameters.tools
            if request.parameters.tool_choice is not None:
                payload["tool_choice"] = request.parameters.tool_choice
            if stream and request.parameters.include_usage is not None:
                payload["stream_options"] = {
                    "include_usage": request.parameters.include_usage
                }  # type ignore

        # Add extra_body if there are provider-specific parameters
        if extra_body:
            payload["extra_body"] = extra_body

        return payload

    async def generate(self, request: ProviderLLMRequest) -> ProviderLLMResponse:
        """Generate using litellm's completion API"""
        payload = self._build_chat_payload(request)

        try:
            logger.info("Request payload", payload=payload)
            start_time = time.perf_counter()
            response = await acompletion(**payload)
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(f"Response received in {latency_ms}", response=response)
            return self._parse_response(response, request.model, latency_ms=latency_ms)
        except Exception as e:
            logger.error("Error in litellm API request", error=str(e), exc_info=True)
            self._handle_error(e)

    async def stream(
        self, request: ProviderLLMRequest
    ) -> AsyncGenerator[ProviderLLMResponseDelta, None]:
        """Generate a streaming response using litellm"""
        payload = self._build_chat_payload(request, stream=True)

        logger.debug("Request payload", payload=payload)

        try:
            start_time = time.perf_counter()
            stream: AsyncIterator = await acompletion(**payload)
            async for chunk in stream:
                # logger.debug("Received chunk", chunk=chunk)
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                # Calculate latency
                latency_ms = None
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    latency_ms = (time.perf_counter() - start_time) * 1000

                metadata = {
                    "response_id": getattr(chunk, "id", None),
                    "created": getattr(chunk, "created", None),
                    "model": getattr(chunk, "model", payload.get("model")),
                    "usage": chunk.usage.model_dump()
                    if hasattr(chunk, "usage") and hasattr(chunk.usage, "model_dump")
                    else None,
                }
                if latency_ms is not None:
                    metadata["latency_ms"] = latency_ms

                parsed_tool_calls = None
                if delta.tool_calls:
                    parsed_tool_calls = [
                        self._parse_tool_call(tc) for tc in delta.tool_calls
                    ]

                yield ProviderLLMResponseDelta(
                    content=delta.content or "",
                    role=getattr(delta, "role", None),
                    tool_calls=parsed_tool_calls,
                    finish_reason=choice.finish_reason,
                    metadata=metadata,
                )
        except Exception as e:
            logger.error(
                "Error in litellm streaming request", error=str(e), exc_info=True
            )
            self._handle_error(e)

    def _parse_response(
        self, response, model: str, latency_ms: float | None = None
    ) -> ProviderLLMResponse:
        """Parse litellm response, handling both standard and provider-specific formats."""
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

            # Extract reasoning content if present
            reasoning_text = None
            if hasattr(message, "reasoning"):
                reasoning_text = getattr(message, "reasoning", None)

            metadata = {
                "response_id": getattr(response, "id", None),
                "created": getattr(response, "created", None),
                "model": getattr(response, "model", model),
                "reasoning": reasoning_text,
            }
            if latency_ms is not None:
                metadata["latency_ms"] = latency_ms

            return ProviderLLMResponse(
                content=message.content or "",
                model=model,
                tokens_used=self._calculate_tokens(response),
                finish_reason=choice.finish_reason,
                tool_calls=tool_calls or None,
                usage=response.usage.model_dump()
                if hasattr(response, "usage") and hasattr(response.usage, "model_dump")
                else None,
                metadata=metadata,
            )
        except Exception as e:
            logger.error("Error parsing response", error=str(e), exc_info=True)
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

            extra_content = tc.get("extra_content")

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

            result = {
                "id": str(tc.get("id", f"call_{id(tool_call)}")),
                "type": str(tc.get("type", "function")),
                "function": {
                    "name": str(function_data.get("name", "")),
                    "arguments": str(function_data.get("arguments", "{}")),
                },
            }
            if extra_content:
                result["extra_content"] = extra_content
            # Build the result
            return result
        except Exception as err:
            logger.warning("Failed to parse tool call", error=str(err), exc_info=True)
            return {
                "id": f"error_{id(err)}",
                "type": "function",
                "function": {
                    "name": "error",
                    "arguments": f'{{ "error": "{str(err).replace("'", "\\'")}" }}',
                },
            }

    def _calculate_tokens(self, response) -> int | None:
        """Calculate total tokens used from the response.

        Args:
            response: The response object from the API

        Returns:
            The total number of tokens used, or None if not available
        """
        if not hasattr(response, "usage"):
            return None

        usage = response.usage

        # Handle case where usage is a dictionary
        if isinstance(usage, Mapping):
            total = usage.get("total_tokens")
            return int(total) if total is not None else None

        # Handle case where usage is an object with attributes
        if hasattr(usage, "total_tokens"):
            total = usage.total_tokens
            return int(total) if total is not None else None

        return None

    def _format_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Format messages for Chat Completions API."""
        import json

        formatted = []
        for msg in messages:
            msg_copy = msg.copy()
            # Convert tool message content from list to string
            if msg_copy.get("role") == "tool" and isinstance(
                msg_copy.get("content"), list
            ):
                text_parts = []
                for item in msg_copy["content"]:
                    if (
                        isinstance(item, dict)
                        and item.get("type") in ["text", "input_text"]
                        and "text" in item
                    ):
                        text_parts.append(item["text"])
                msg_copy["content"] = "".join(text_parts)
            if msg_copy.get("tool_calls"):
                # Ensure arguments are JSON strings
                for tool_call in msg_copy["tool_calls"]:
                    if isinstance(tool_call.get("function", {}).get("arguments"), dict):
                        tool_call["function"]["arguments"] = json.dumps(
                            tool_call["function"]["arguments"]
                        )
            formatted.append(msg_copy)
        return formatted

    async def health_check(self) -> bool:
        """Check if the API is accessible using litellm's get_valid_models"""
        # Set global litellm configuration
        litellm.api_base = self.base_url
        litellm.api_key = self.api_key

        try:
            # Query the live active models endpoint using 'openai' as the target format
            models = get_valid_models(
                check_provider_endpoint=True, custom_llm_provider="openai"
            )
            return bool(models)
        except Exception:
            return False

    def _extract_error_details(self, e: Exception) -> tuple[int | None, Any]:
        """Extract status code and response from exception.

        Args:
            e: The exception to extract details from

        Returns:
            Tuple of (status_code, response) where either or both can be None
        """
        # Handle case where status_code is directly on the exception
        if hasattr(e, "status_code"):
            status_code = e.status_code
            if status_code is None:
                return None, getattr(e, "response", None)
            try:
                return int(str(status_code)), getattr(e, "response", None)
            except (ValueError, TypeError):
                return None, getattr(e, "response", None)

        # Handle case where status_code is in the response object
        if hasattr(e, "response") and hasattr(e.response, "status_code"):
            status_code = e.response.status_code
            if status_code is None:
                return None, e.response
            try:
                return int(str(status_code)), e.response
            except (ValueError, TypeError):
                return None, e.response

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
    """Driver for OpenAI responses API using litellm"""

    def __init__(self, api_key: str | None, base_url: str, **kwargs):
        # Ensure base_url doesn't end with /responses
        base_url = base_url.rstrip("/")
        base_url = base_url.removesuffix("/responses")

        super().__init__(api_key, base_url, **kwargs)
        self.provider_name = kwargs.get("provider_name")

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
                logger.warning("Unsupported tool format", tool=tool)
                formatted_tools.append(tool)

        return formatted_tools

    def _format_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Format messages for Responses API input structure."""
        formatted = []
        for msg in messages:
            if msg.get("role") == "user":
                formatted.append(
                    {
                        "type": "message",
                        "role": "user",
                        "content": msg.get("content", ""),
                    }
                )
            elif msg.get("role") == "assistant":
                item = {
                    "type": "message",
                    "role": "assistant",
                    "content": msg.get("content", None),
                }
                if msg.get("tool_calls"):
                    item["tool_calls"] = msg["tool_calls"]
                formatted.append(item)
            elif msg.get("role") == "tool":
                # Extract text from content if it's a list
                content = msg.get("content", "")
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if (
                            item.get("type") in ["text", "input_text"]
                            and "text" in item
                        ):
                            text_parts.append(item["text"])
                    output = "".join(text_parts)
                else:
                    output = str(content)
                formatted.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg.get("tool_call_id", ""),
                        "output": output,
                    }
                )
            # Skip system messages, they go to instructions
        return formatted

    def _build_responses_payload(  # noqa C901
        self, request: ProviderLLMRequest, stream: bool = False
    ) -> dict[str, Any]:
        """Build payload for responses API using litellm format."""
        # Use "openai/responses/<model_name>" to tell litellm to use Responses API schema
        model = f"openai/responses/{request.model}"

        # Extract system messages for instructions
        system_messages = [
            msg for msg in request.messages if msg.get("role") == "system"
        ]
        instructions = "\n".join([msg.get("content", "") for msg in system_messages])

        payload: dict[str, Any] = {
            "model": model,
            "input": self._format_messages(request.messages),
            "temperature": request.temperature,
        }
        if stream:
            payload["stream"] = True

        # Add base_url and api_key for litellm
        payload["api_base"] = self.base_url
        payload["api_key"] = self.api_key

        # Handle provider-specific parameters via extra_body
        extra_body = {}
        if request.parameters:
            # Check for extra_body.thinking_config for Gemini
            if (
                hasattr(request.parameters, "extra_body")
                and request.parameters.extra_body
            ):
                extra_body.update(request.parameters.extra_body)

        # Add instructions if present
        if instructions:
            payload["instructions"] = instructions

        # Add parameters explicitly
        if request.parameters:
            if request.parameters.max_tokens is not None:
                payload["max_output_tokens"] = request.parameters.max_tokens
            if request.parameters.top_p is not None:
                payload["top_p"] = request.parameters.top_p
            if request.parameters.frequency_penalty is not None:
                payload["frequency_penalty"] = request.parameters.frequency_penalty
            if request.parameters.presence_penalty is not None:
                payload["presence_penalty"] = request.parameters.presence_penalty
            if request.parameters.seed is not None:
                payload["seed"] = request.parameters.seed
            if request.parameters.stop is not None:
                payload["stop"] = request.parameters.stop
            if request.parameters.response_format is not None:
                payload["response_format"] = request.parameters.response_format
            if request.parameters.tools is not None:
                payload["tools"] = self._format_tools_for_responses_api(
                    request.parameters.tools
                )
            if request.parameters.tool_choice is not None:
                payload["tool_choice"] = request.parameters.tool_choice

        # Handle reasoning effort
        if request.parameters and request.parameters.reasoning_effort is not None:
            payload["reasoning"] = {"effort": request.parameters.reasoning_effort}
        else:
            payload["reasoning"] = {"effort": "low"}

        # Format tools if present
        if "tools" in payload:
            payload["tools"] = self._format_tools_for_responses_api(payload["tools"])
            if "tool_choice" not in payload:
                payload["tool_choice"] = "auto"

        # Add extra_body if there are provider-specific parameters
        if extra_body:
            payload["extra_body"] = extra_body

        return payload

    async def generate(self, request: ProviderLLMRequest) -> ProviderLLMResponse:
        """Generate using litellm's completion API with responses schema"""
        payload = self._build_responses_payload(request)

        logger.debug("Request payload", payload=payload)
        try:
            start_time = time.perf_counter()
            response = await acompletion(**payload)
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(f"Response received in {latency_ms}", response=response)
            return self._parse_response(response, request.model, latency_ms=latency_ms)
        except Exception as e:
            self._handle_error(e)

    async def stream(
        self, request: ProviderLLMRequest
    ) -> AsyncGenerator[ProviderLLMResponseDelta, None]:
        """Generate a streaming response using litellm"""
        payload = self._build_responses_payload(request, stream=True)

        logger.debug("Request payload", payload=payload)
        try:
            start_time = time.perf_counter()
            stream = await acompletion(**payload)
            normalizer = ResponseNormalizer(start_time)
            async for delta in normalizer.normalize_stream(stream):
                yield delta
        except Exception as e:
            self._handle_error(e)

    def _extract_usage(self, response) -> dict[str, Any] | None:
        """Extract usage data from ResponseUsage object"""
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
        return usage

    def _extract_text_from_content_list(self, content: list) -> str | None:
        """Extract text from a list of content items."""
        texts = []
        for p in content:
            text = ""
            if isinstance(p, dict):
                text = p.get("text")
            elif hasattr(p, "text"):
                text = p.text
            if text:
                texts.append(str(text))
        return "".join(texts) if texts else None

    def _extract_reasoning(self, response) -> str | None:
        """Extraction of reasoning text."""
        output = getattr(response, "output", None)
        if not output or not isinstance(output, list):
            return None

        for item in output:
            item_type = (
                item.get("type")
                if isinstance(item, dict)
                else getattr(item, "type", None)
            )
            if item_type in {"reasoning", "reasoning_text"}:
                content = (
                    item.get("content") if isinstance(item, dict) else item.content
                )
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    return self._extract_text_from_content_list(content)
        return None

    def _build_metadata(
        self, response, reasoning_text: str | None, latency_ms: float | None
    ) -> dict[str, Any]:
        """Build metadata dictionary for the response"""
        metadata = {
            "response_id": getattr(response, "id", None),
            "created": getattr(response, "created_at", None),
            "model": getattr(response, "model", None),
            "reasoning": reasoning_text,
        }
        if latency_ms is not None:
            metadata["latency_ms"] = latency_ms
        return metadata

    def _parse_response(
        self, response: Any, model: str, latency_ms: float | None = None
    ) -> ProviderLLMResponse:
        """Parse OpenAI responses API response"""
        usage = self._extract_usage(response)
        reasoning_text = self._extract_reasoning(response)
        tool_calls = (
            self._parse_tool_calls(response) if hasattr(response, "output") else None
        )
        metadata = self._build_metadata(response, reasoning_text, latency_ms)

        parsed_response = ProviderLLMResponse(
            content=response.output_text,
            model=model,
            tokens_used=usage.get("total_tokens") if isinstance(usage, dict) else None,
            finish_reason=getattr(response, "finish_reason", None),
            tool_calls=tool_calls,
            usage=usage,
            reasoning=reasoning_text,
            metadata=metadata,
        )
        logger.debug("Response parsed", parsed_response=parsed_response)
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

        output = getattr(response, "output", None)
        if output is not None:
            # Convert ResponseUsage to dict if it's not already
            if isinstance(output, list):
                for output_item in output:
                    # Parse the tool call
                    item_type = (
                        output_item.get("type")
                        if isinstance(output_item, dict)
                        else getattr(output_item, "type", None)
                    )
                    if item_type == "function_call":
                        try:
                            # Extract provider-specific fields from output item
                            extra_content = output_item.get("extra_content")

                            # Parse the arguments if it's a string - handle both dict and object types
                            # For Responses API, the function call has a "function" subdict
                            function_data = (
                                output_item.get("function", {})
                                if isinstance(output_item, dict)
                                else getattr(output_item, "function", {})
                            )
                            if isinstance(function_data, dict):
                                arguments = function_data.get("arguments", "{}")
                                name = function_data.get("name", "")
                            else:
                                # Fallback for object attributes
                                arguments = (
                                    getattr(function_data, "arguments", "{}")
                                    if hasattr(function_data, "arguments")
                                    else "{}"
                                )
                                name = (
                                    getattr(function_data, "name", "")
                                    if hasattr(function_data, "name")
                                    else ""
                                )

                            if isinstance(arguments, str):
                                arguments = json.loads(arguments)

                            tool_call = {
                                "id": (
                                    output_item.get("call_id")
                                    if isinstance(output_item, dict)
                                    else getattr(output_item, "call_id", None)
                                )
                                or (
                                    output_item.get("id")
                                    if isinstance(output_item, dict)
                                    else getattr(output_item, "id", None)
                                )
                                or f"call_{id(output_item)}",
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": arguments,
                                },
                            }

                            # Add provider-specific fields if present
                            if extra_content:
                                tool_call["extra_content"] = extra_content

                            tool_calls.append(tool_call)
                        except (json.JSONDecodeError, AttributeError) as e:
                            logger.warning(
                                "Failed to parse tool call arguments",
                                error=str(e),
                                exc_info=True,
                            )
                            continue

        return tool_calls

    async def health_check(self) -> bool:
        """Check if the API is accessible using litellm's get_valid_models"""
        # Set global litellm configuration
        litellm.api_base = self.base_url
        litellm.api_key = self.api_key

        try:
            # Query the live active models endpoint using 'openai' as the target format
            models = get_valid_models(
                check_provider_endpoint=True, custom_llm_provider="openai"
            )
            return bool(models)
        except Exception:
            return False

    def _extract_error_details(self, e: Exception) -> tuple[int | None, Any]:
        """Extract status code and response from exception.

        Args:
            e: The exception to extract details from

        Returns:
            Tuple of (status_code, response) where either or both can be None
        """
        # Handle case where status_code is directly on the exception
        if hasattr(e, "status_code"):
            status_code = e.status_code
            if status_code is None:
                return None, getattr(e, "response", None)
            try:
                return int(str(status_code)), getattr(e, "response", None)
            except (ValueError, TypeError):
                return None, getattr(e, "response", None)

        # Handle case where status_code is in the response object
        if hasattr(e, "response") and hasattr(e.response, "status_code"):
            status_code = e.response.status_code
            if status_code is None:
                return None, e.response
            try:
                return int(str(status_code)), e.response
            except (ValueError, TypeError):
                return None, e.response
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


class MockDriver(BaseDriver):
    """Driver for testing with pregenerated LLM responses from files.

    This driver loads pre-canned responses from JSON files instead of making
    real API calls, enabling manual testing without LLM costs or latency.

    Response files should be JSON files with the following structure:
    {
        "request_pattern": "optional pattern to match requests",
        "response": {
            "content": "pre-generated response text",
            "model": "model-name",
            "tokens_used": 100,
            "finish_reason": "stop",
            "tool_calls": null,
            "usage": {...},
            "reasoning": null,
            "metadata": {...}
        },
        "streaming_chunks": [
            {"content": "chunk1", "finish_reason": null},
            {"content": "chunk2", "finish_reason": "stop"}
        ]
    }
    """

    def __init__(self, api_key: str | None, base_url: str, **kwargs):
        super().__init__(api_key, base_url, **kwargs)
        responses_dir = kwargs.get("responses_dir") or kwargs.get("mock_responses_dir")
        self.default_response = kwargs.get("default_response")
        self._response_cache: dict[str, dict] = {}

        logger.debug(
            "MockDriver initialized",
            kwargs_keys=list(kwargs.keys()),
            responses_dir=responses_dir,
        )

        # Resolve path alias if present
        if (
            responses_dir
            and isinstance(responses_dir, str)
            and responses_dir.startswith("@")
        ):
            try:
                from local_coding_assistant.config.env_manager import get_env_manager

                env_manager = get_env_manager()
                path_manager = getattr(env_manager, "path_manager", None)
                if not path_manager:
                    raise ValueError("Path manager not found in env manager")
                else:
                    self.responses_dir = str(path_manager.resolve_path(responses_dir))
                    logger.debug(
                        f"Resolved mock_responses_dir to: {self.responses_dir}"
                    )
            except Exception as e:
                logger.debug(f"Failed to resolve path alias {responses_dir}: {e}")
                self.responses_dir = responses_dir
        else:
            self.responses_dir = responses_dir

    def _load_responses(self) -> dict[str, dict]:
        """Load all response JSON files from the responses directory."""
        if self._response_cache:
            return self._response_cache

        if not self.responses_dir:
            logger.debug("No responses_dir configured for MockDriver")
            return {}

        from pathlib import Path

        responses_path = Path(self.responses_dir)
        if not responses_path.exists():
            logger.debug(f"Responses directory does not exist: {self.responses_dir}")
            return {}

        responses = {}
        for json_file in responses_path.glob("*.json"):
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)
                    # Use filename as key if no pattern specified
                    key = data.get("request_pattern", json_file.stem)
                    responses[key] = data
            except Exception as e:
                logger.debug(f"Failed to load response file {json_file}: {e}")

        self._response_cache = responses
        return responses

    def _generate_request_key(self, request: ProviderLLMRequest) -> str:
        """Generate a key for matching the request to a pregenerated response."""
        # Simple hash based on model and last user message content
        import hashlib

        user_messages = [msg for msg in request.messages if msg.get("role") == "user"]
        if user_messages:
            content = user_messages[-1].get("content", "")
        else:
            content = ""

        key_data = f"{request.model}:{content[:200]}"  # First 200 chars
        return hashlib.md5(key_data.encode()).hexdigest()  # noqa S324

    def _find_matching_response(
        self, request: ProviderLLMRequest, responses: dict[str, dict]
    ) -> dict | None:
        """Find a matching pregenerated response for the request."""
        request_key = self._generate_request_key(request)

        # Try exact match by hash
        if request_key in responses:
            return responses[request_key]

        # Try pattern matching on request content
        user_messages = [msg for msg in request.messages if msg.get("role") == "user"]
        if user_messages:
            content = user_messages[-1].get("content", "")
            for pattern, response_data in responses.items():
                if pattern and pattern.lower() in content.lower():
                    logger.debug(f"Matched response by pattern: {pattern}")
                    return response_data

        # Try model-based fallback
        model_key = request.model
        if model_key in responses:
            return responses[model_key]

        # Use default if available
        if self.default_response:
            return {"response": self.default_response}

        return None

    async def generate(self, request: ProviderLLMRequest) -> ProviderLLMResponse:
        """Generate a response using preloaded mock data."""
        logger.info("Request messages", messages=request.messages)
        responses = self._load_responses()
        response_data = self._find_matching_response(request, responses)

        if not response_data:
            logger.debug(
                "No matching pregenerated response found, using fallback",
                request_model=request.model,
            )
            # Return a generic fallback response
            return ProviderLLMResponse(
                content="[Mock response: No pregenerated response found for this request]",
                model=request.model,
                tokens_used=0,
                finish_reason="stop",
                metadata={"mock": True, "fallback": True},
            )

        response_obj = response_data.get("response", {})
        logger.debug(
            "Returning pregenerated response",
            response_keys=list(response_obj.keys()),
        )

        return ProviderLLMResponse(
            content=response_obj.get("content", ""),
            model=response_obj.get("model", request.model),
            tokens_used=response_obj.get("tokens_used"),
            finish_reason=response_obj.get("finish_reason"),
            tool_calls=response_obj.get("tool_calls"),
            usage=response_obj.get("usage"),
            reasoning=response_obj.get("reasoning"),
            metadata={**response_obj.get("metadata", {}), "mock": True},
        )

    async def stream(
        self, request: ProviderLLMRequest
    ) -> AsyncGenerator[ProviderLLMResponseDelta, None]:
        """Generate a streaming response using preloaded mock data."""
        logger.debug("Request messages", messages=request.messages)
        responses = self._load_responses()
        response_data = self._find_matching_response(request, responses)

        if not response_data or "streaming_chunks" not in response_data:
            # Fallback to non-streaming response as single delta
            response = await self.generate(request)
            yield ProviderLLMResponseDelta(
                content=response.content,
                finish_reason=response.finish_reason,
                metadata=response.metadata,
            )
            return

        chunks = response_data.get("streaming_chunks", [])
        for chunk in chunks:
            yield ProviderLLMResponseDelta(
                content=chunk.get("content", ""),
                finish_reason=chunk.get("finish_reason"),
                reasoning=chunk.get("reasoning"),
                tool_calls=chunk.get("tool_calls"),
                metadata={**chunk.get("metadata", {}), "mock": True},
            )

    async def health_check(self) -> bool:
        """Check if mock responses are available."""
        responses = self._load_responses()
        return bool(responses)
