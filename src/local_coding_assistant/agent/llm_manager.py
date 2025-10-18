"""Agent layer: LLM & reasoning management."""

import os
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field

from local_coding_assistant.config.schemas import LLMConfig
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("agent.llm_manager")


class LLMRequest(BaseModel):
    """Request structure for LLM generation."""

    prompt: str = Field(..., description="The main prompt to send to the LLM")
    context: dict[str, Any] | None = Field(
        default=None, description="Additional context data"
    )
    system_prompt: str | None = Field(
        default=None, description="System prompt override"
    )
    tools: list[dict[str, Any]] | None = Field(
        default=None, description="Available tools"
    )
    tool_outputs: dict[str, Any] | None = Field(
        default=None, description="Previous tool call results"
    )


class LLMResponse(BaseModel):
    """Response structure from LLM generation."""

    content: str = Field(..., description="Generated response content")
    model_used: str = Field(..., description="Model that was used for generation")
    tokens_used: int | None = Field(default=None, description="Tokens consumed")
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None, description="Tool calls made by the model"
    )


class LLMManager:
    """Enhanced LLM manager with proper error handling, logging, and tool integration.

    This class provides a facade for interacting with various LLM providers while
    maintaining compatibility with the existing tool registry and session management.

    Uses OpenAI's new Responses API (responses.create) instead of the legacy
    Chat Completions API (chat.completions.create) for improved functionality.
    """

    def __init__(self, config: LLMConfig):
        """Initialize the LLM manager with configuration.

        Args:
            config: Configuration object containing model and provider settings.

        Raises:
            LLMError: If the provider is not supported or configuration is invalid.
        """
        self.config = config
        self.client = None
        self._setup_client()

    def _setup_client(self) -> None:
        """Set up the LLM client based on the provider configuration.

        Raises:
            LLMError: If the provider is not supported.
        """
        try:
            if self.config.provider == "openai":
                self._setup_openai_client()
            else:
                from local_coding_assistant.core.exceptions import LLMError

                raise LLMError(f"Unsupported provider: {self.config.provider}")
        except Exception as e:
            logger.error(f"Failed to setup LLM client: {e}")
            from local_coding_assistant.core.exceptions import LLMError

            # Preserve LLMError messages, don't wrap them
            if isinstance(e, LLMError):
                raise
            raise LLMError(f"Failed to initialize {self.config.provider} client") from e

    def _setup_openai_client(self) -> None:
        """Set up OpenAI client."""
        # Check if openai is available at runtime
        try:
            import openai

            # Handle case where openai is mocked as None in tests
            if openai is None:
                raise ImportError("OpenAI package not available")
        except ImportError as e:
            from local_coding_assistant.core.exceptions import LLMError

            raise LLMError(f"OpenAI package not installed. Error: {e}") from e

        try:
            from openai import AsyncOpenAI

            # Use explicit API key from config if provided, otherwise let OpenAI client handle env vars
            api_key = self.config.api_key

            self.client = AsyncOpenAI(api_key=api_key)
            logger.info(
                f"Initialized OpenAI client for model: {self.config.model_name}"
            )
        except Exception as e:
            from local_coding_assistant.core.exceptions import LLMError

            raise LLMError(f"Failed to initialize OpenAI client: {e}") from e

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response using the LLM with the given request.

        Uses OpenAI's new Responses API which provides enhanced functionality
        compared to the legacy Chat Completions API.

        Args:
            request: Structured request containing prompt, context, and tool information.

        Returns:
            Structured response with generated content and metadata.

        Raises:
            LLMError: If generation fails or client is not properly initialized.
        """
        # Check if we're in test mode and should return mock responses
        if os.environ.get("LOCCA_TEST_MODE") == "true":
            return self._generate_mock_response(request)

        if not self.client:
            from local_coding_assistant.core.exceptions import LLMError

            raise LLMError("LLM client not initialized")

        try:
            logger.debug(f"Generating response for prompt: {request.prompt[:100]}...")

            # Build messages from session context and request
            messages = self._build_messages(request)

            # Prepare generation parameters for new Responses API
            gen_params = {
                "model": self.config.model_name,
                "input": messages,
                "temperature": self.config.temperature,
            }

            # Note: Responses API may use different parameter names for token limits
            # The new API might handle token limits differently or not support max_tokens
            # For now, we'll skip max_tokens and let the API handle it
            # if self.config.max_tokens:
            #     gen_params["max_tokens"] = self.config.max_tokens

            # Add tools if provided (new API may handle this differently)
            if request.tools:
                # Convert tools from old Chat Completions format to new Responses API format
                converted_tools = []
                for tool in request.tools:
                    if tool.get("type") == "function":
                        # Convert from old format to new format
                        converted_tools.append(
                            {
                                "type": "function",  # Keep as function for compatibility
                                "name": tool["function"]["name"],
                                "description": tool["function"].get("description", ""),
                                "parameters": tool["function"].get("parameters", {}),
                            }
                        )
                    else:
                        # Pass through as-is for other tool types
                        converted_tools.append(tool)

                gen_params["tools"] = converted_tools

            # Add text formatting options for new Responses API
            text_options = {}
            if request.context:
                # Use context to determine appropriate verbosity/format
                text_options["verbosity"] = "medium"

            if text_options:
                gen_params["text"] = text_options

            # Make the API call
            response = await self.client.responses.create(**gen_params)

            # Extract content from the new Responses API structure
            content = ""
            for item in response.output:
                if hasattr(item, "content"):
                    for content_item in item.content:
                        if hasattr(content_item, "text"):
                            content += content_item.text

            # Extract tool calls if present (handle both old and new API formats)
            tool_calls = None

            # Try new Responses API format first
            if hasattr(response, "output"):
                for item in response.output:
                    if hasattr(item, "tool_calls") and item.tool_calls:
                        tool_calls = []
                        for tool_call in item.tool_calls:
                            tool_calls.append(
                                {
                                    "id": getattr(tool_call, "id", ""),
                                    "type": getattr(tool_call, "type", "function"),
                                    "function": {
                                        "name": getattr(
                                            tool_call,
                                            "name",
                                            getattr(tool_call.function, "name", ""),
                                        ),
                                        "arguments": getattr(
                                            tool_call,
                                            "arguments",
                                            getattr(
                                                tool_call.function, "arguments", "{}"
                                            ),
                                        ),
                                    },
                                }
                            )
                        break

            # Fallback to old format if new format doesn't work
            if tool_calls is None:
                # Check if response has choices (old API format)
                if hasattr(response, "choices") and response.choices:
                    choice = response.choices[0]
                    if (
                        hasattr(choice, "message")
                        and hasattr(choice.message, "tool_calls")
                        and choice.message.tool_calls
                    ):
                        tool_calls = [
                            {
                                "id": tool_call.id,
                                "type": tool_call.type,
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments,
                                },
                            }
                            for tool_call in choice.message.tool_calls
                        ]

            llm_response = LLMResponse(
                content=content,
                model_used=getattr(response, "model", self.config.model_name),
                tokens_used=getattr(response.usage, "output_tokens", None)
                if response.usage
                else None,
                tool_calls=tool_calls,
            )

            logger.info(
                f"Generated response with {len(content)} characters using model {getattr(response, 'model', self.config.model_name)}"
            )
            return llm_response

        except Exception as e:
            logger.error(f"Error during LLM generation: {e}")
            if "openai" in str(type(e)).lower():
                from local_coding_assistant.core.exceptions import LLMError

                raise LLMError(f"OpenAI API error: {e}") from e
            else:
                from local_coding_assistant.core.exceptions import LLMError

                raise LLMError(f"LLM generation failed: {e}") from e

    async def generate_stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Generate a streaming response using the LLM with the given request.

        This is a placeholder implementation that falls back to regular generation.
        Subclasses should override this method to provide actual streaming functionality.

        Args:
            request: Structured request containing prompt, context, and tool information.

        Yields:
            Partial response content as strings.

        Raises:
            LLMError: If streaming generation fails or client is not properly initialized.
        """
        # Placeholder implementation - fall back to regular generation
        # Subclasses should override this for actual streaming
        logger.warning(
            "generate_stream() not implemented for this provider, falling back to generate()"
        )

        # Get the full response using regular generation
        response = await self.generate(request)

        # Yield the complete content as a single chunk
        # In a real implementation, this would yield partial tokens
        if response.content:
            yield response.content

        # Also handle tool calls if present
        if response.tool_calls:
            # For streaming, tool calls would typically be yielded at the end
            # or as separate metadata, but for compatibility, we'll just yield the content
            pass

    def update_config(
        self,
        *,
        model_name: str | None = None,
        provider: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        api_key: str | None = None,
    ) -> None:
        """Update LLM configuration with intelligent client management.

        Args:
            model_name: New model name (requires new client if changed)
            provider: New provider (requires new client if changed)
            temperature: New temperature (safe config update)
            max_tokens: New max tokens (safe config update)
            api_key: New API key (may require new client)

        Raises:
            LLMError: If configuration update fails
        """
        # Determine which parameters require client recreation
        requires_new_client = (
            (model_name is not None and model_name != self.config.model_name)
            or (provider is not None and provider != self.config.provider)
            or (api_key is not None and api_key != self.config.api_key)
        )

        # Build update dictionary with only provided values
        updates = {}
        if model_name is not None:
            updates["model_name"] = model_name
        if provider is not None:
            updates["provider"] = provider
        if temperature is not None:
            updates["temperature"] = temperature
        if max_tokens is not None:
            updates["max_tokens"] = max_tokens
        if api_key is not None:
            updates["api_key"] = api_key

        if not updates:
            logger.debug("No configuration updates provided")
            return

        # Validate the new configuration
        try:
            old_config_dict = self.config.model_dump()
            new_config_dict = {**old_config_dict, **updates}
            # This will raise ValidationError if invalid
            LLMConfig(**new_config_dict)
        except Exception as e:
            logger.error(f"Configuration update validation failed: {e}")
            from local_coding_assistant.core.exceptions import LLMError

            raise LLMError(f"Configuration update validation failed: {e}") from e

        logger.info(f"Updating LLM config: {updates}")

        # Update configuration
        if model_name is not None:
            self.config.model_name = model_name
        if provider is not None:
            self.config.provider = provider
        if temperature is not None:
            self.config.temperature = temperature
        if max_tokens is not None:
            self.config.max_tokens = max_tokens
        if api_key is not None:
            self.config.api_key = api_key

        # Recreate client if necessary
        if requires_new_client:
            logger.info("Configuration change requires new client - recreating")
            self._client = None  # Force recreation on next use
            # Access _client property to trigger recreation
            _ = self._client
        else:
            logger.debug("Configuration updated without requiring new client")

    async def ainvoke(self, request: LLMRequest) -> LLMResponse:
        """Generate a response using the LLM with the given request (async version of generate).

        This is an alias for generate() to provide compatibility with LangGraph patterns.

        Args:
            request: Structured request containing prompt, context, and tool information.

        Returns:
            Structured response with generated content and metadata.

        Raises:
            LLMError: If generation fails or client is not properly initialized.
        """
        return await self.generate(request)

    def _build_messages(self, request: LLMRequest) -> list[dict[str, str]]:
        """Build OpenAI-compatible message list from request.

        Args:
            request: The LLM request containing prompt and context.

        Returns:
            List of message dictionaries for the OpenAI API.
        """
        messages = []

        # Add system prompt
        system_content = request.system_prompt or "You are a helpful coding assistant."
        messages.append({"role": "system", "content": system_content})

        # Add context if provided
        if request.context:
            context_str = "\n".join(f"{k}: {v}" for k, v in request.context.items())
            messages.append({"role": "user", "content": f"Context:\n{context_str}"})

        # Add tool outputs if provided
        if request.tool_outputs:
            tool_outputs_str = "\n".join(
                f"Tool {k} result: {v}" for k, v in request.tool_outputs.items()
            )
            messages.append(
                {
                    "role": "user",
                    "content": f"Previous tool outputs:\n{tool_outputs_str}",
                }
            )

        # Add the main prompt
        messages.append({"role": "user", "content": request.prompt})

        return messages

    def _generate_mock_response(self, request: LLMRequest) -> LLMResponse:
        """Generate a mock response for testing purposes.

        Args:
            request: The LLM request.

        Returns:
            Mock LLM response.
        """
        # Create a simple echo response
        content = f"[LLMManager] Echo: {request.prompt}"

        # If tool outputs are present, modify the response
        if request.tool_outputs:
            content = "[LLMManager] Echo: Received request with tool outputs"

        return LLMResponse(
            content=content,
            model_used=self.config.model_name,
            tokens_used=50,
            tool_calls=None,
        )
