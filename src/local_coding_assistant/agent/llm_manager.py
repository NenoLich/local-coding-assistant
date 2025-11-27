"""
Enhanced LLM Manager with Provider System Integration

This module provides an updated LLM manager that integrates with the new provider system
while maintaining backward compatibility with existing code.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from local_coding_assistant.core.exceptions import LLMError
from local_coding_assistant.utils.logging import get_logger

if TYPE_CHECKING:
    # Type-only imports to avoid circular import at runtime
    from local_coding_assistant.providers import (
        BaseProvider,
        ProviderLLMRequest,
    )

logger = get_logger("agent.llm_manager")


class LLMRequest:
    """Request structure for LLM generation."""

    def __init__(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_outputs: dict[str, Any] | None = None,
    ):
        self.prompt = prompt
        self.context = context or {}
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.tool_outputs = tool_outputs or {}

    def to_provider_request(self, model: str) -> ProviderLLMRequest:
        """Convert to provider request format."""
        # Import the ProviderLLMRequest class at runtime to avoid circular imports
        from local_coding_assistant.providers.base import (
            OptionalParameters,
            ProviderLLMRequest,
        )

        messages = []

        # Add system prompt
        self._add_system_prompt(messages)

        # Add history messages from context if provided
        self._add_history_messages(messages)

        # Add context if provided
        self._add_context_messages(messages)

        # Add tool outputs if provided
        self._add_tool_outputs_messages(messages)

        # Add the main prompt
        self._add_main_prompt(messages)

        # Create optional parameters
        parameters = OptionalParameters(
            stream=False,
            tools=self.tools,  # self.tools is already a list with default_factory=list in LLMRequest
            tool_choice="auto" if self.tools else None,
            response_format=None,
            max_tokens=None,
        )

        return ProviderLLMRequest(
            messages=messages,
            model=model,
            temperature=0.7,  # Default temperature
            tool_outputs=self.tool_outputs if self.tool_outputs else None,
            parameters=parameters,
        )

    def _add_system_prompt(self, messages: list[dict[str, Any]]) -> None:
        """Add system prompt to messages."""
        system_content = self.system_prompt or "You are a helpful coding assistant."
        messages.append({"role": "system", "content": system_content})

    def _add_history_messages(self, messages: list[dict[str, Any]]) -> None:
        """Add history messages from context to messages."""
        if self.context and "history" in self.context:
            history = self.context.get("history", [])
            for msg in history:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    messages.append({"role": msg["role"], "content": msg["content"]})

    def _add_context_messages(self, messages: list[dict[str, Any]]) -> None:
        """Add context messages to messages."""
        if self.context:
            context_str = "\n".join(f"{k}: {v}" for k, v in self.context.items())

            # Check if the prompt is already the last message in history to avoid duplication
            history = self.context.get("history", [])
            should_add_context = True

            if history and self.prompt:
                # Get the last message from history
                last_history_message = history[-1]
                if (
                    isinstance(last_history_message, dict)
                    and last_history_message.get("role") == "user"
                    and last_history_message.get("content") == self.prompt
                ):
                    # Context already contains the current prompt, don't add it again
                    should_add_context = False

            if should_add_context:
                messages.append({"role": "user", "content": f"Context:\n{context_str}"})

    def _add_tool_outputs_messages(self, messages: list[dict[str, Any]]) -> None:
        """Add tool outputs messages to messages."""
        if self.tool_outputs:
            tool_outputs_str = "\n".join(
                f"Tool {k} result: {v}" for k, v in self.tool_outputs.items()
            )
            messages.append(
                {
                    "role": "user",
                    "content": f"Previous tool outputs:\n{tool_outputs_str}",
                }
            )

    def _add_main_prompt(self, messages: list[dict[str, Any]]) -> None:
        """Add the main prompt to messages."""
        # Check if the prompt is already the last message in the history
        if not messages or not (
            messages[-1].get("role") == "user"
            and messages[-1].get("content") == self.prompt
        ):
            messages.append({"role": "user", "content": self.prompt})


class LLMResponse:
    """Response structure from LLM generation."""

    def __init__(
        self,
        content: str,
        model_used: str,
        tokens_used: int | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ):
        self.content = content
        self.model_used = model_used
        self.tokens_used = tokens_used
        self.tool_calls = tool_calls


class LLMManager:
    """Enhanced LLM manager with provider system integration."""

    def __init__(self, config_manager, provider_manager=None):
        """Initialize the LLM manager with a config manager.

        Args:
            config_manager: An instance of ConfigManager
            provider_manager: Optional instance of ProviderManager
        """
        if config_manager is None:
            raise ValueError("config_manager is required")

        # Import provider classes lazily at runtime to avoid circular imports
        from local_coding_assistant.providers import ProviderManager, ProviderRouter

        self.config_manager = config_manager
        self.provider_manager = provider_manager or ProviderManager(
            env_manager=config_manager.env_manager
        )
        self.router = ProviderRouter(self.config_manager, self.provider_manager)
        self._current_provider = None
        self._current_model = None
        self._background_tasks = []

        # Provider status cache
        self._provider_status_cache: dict[str, dict[str, Any]] = {}
        self._last_health_check: float = 0
        self._cache_ttl: float = 30.0 * 60.0  # 30 minutes cache TTL

        # Initialize provider system with config
        self.provider_manager.reload(self.config_manager)

    async def generate(
        self,
        request: LLMRequest,
        *,
        provider: str | None = None,
        model: str | None = None,
        policy: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Generate a response using the provider system with fallback support."""

        try:
            # Route request and handle generation response
            selected_provider, provider_request = await self._route_request(
                request, provider, model, policy, overrides, stream=False
            )

            return await self._handle_generation_response(
                selected_provider, provider_request
            )

        except Exception as e:
            # Handle critical errors with fallback
            return await self._handle_critical_error_generation(
                e, request, provider, model, policy, overrides
            )

    async def ainvoke(self, request: LLMRequest) -> LLMResponse:
        """Async invoke (alias for generate)."""
        return await self.generate(request)

    def _generate_mock_response(self, request: LLMRequest) -> LLMResponse:
        """Generate a mock response for testing."""
        content = f"[LLMManager] Echo: {request.prompt}"

        if request.tool_outputs:
            content = "[LLMManager] Echo: Received request with tool outputs"

        return LLMResponse(
            content=content,
            model_used="mock-model",
            tokens_used=50,
            tool_calls=None,
        )

    async def _route_request(
        self,
        request: LLMRequest,
        provider: str | None = None,
        model: str | None = None,
        policy: str | None = None,
        overrides: dict[str, Any] | None = None,
        stream: bool = True,
    ) -> tuple[BaseProvider, ProviderLLMRequest]:
        """Route request to appropriate provider and convert request format.

        Args:
            request: The LLM request
            provider: Optional provider override
            model: Optional model override
            policy: Optional policy override
            overrides: Optional configuration overrides
            stream: Whether to enable streaming in the request

        Returns:
            Tuple of (selected_provider, provider_request)
        """
        # Apply overrides
        overrides = overrides or {}
        effective_model = model or overrides.get("llm.model_name", None)
        effective_temperature: float | None = overrides.get("llm.temperature", None)
        effective_max_tokens: int | None = overrides.get("llm.max_tokens", None)

        # First convert the request to a provider request with a default model
        # The actual model will be set after routing
        temp_model = (
            effective_model or "default"
        )  # Use 'default' as a fallback model name

        # Create the provider request with streaming disabled initially
        provider_request = request.to_provider_request(temp_model)

        # Initialize parameters if not already set
        if provider_request.parameters is None:
            from local_coding_assistant.providers.base import OptionalParameters

            provider_request.parameters = OptionalParameters()

        # Set streaming in parameters
        provider_request.parameters.stream = stream

        # Set the model if specified
        if effective_model:
            provider_request.model = effective_model

        # Apply temperature and max_tokens overrides if provided
        if effective_temperature is not None:
            provider_request.temperature = effective_temperature

        if effective_max_tokens is not None:
            provider_request.parameters.max_tokens = effective_max_tokens

        # Let the router handle provider and model selection with built-in validation and fallback
        selected_provider, selected_model = await self.router.get_provider_for_request(
            request=provider_request, role=policy, provider=provider
        )

        # Update the provider request with the selected model
        provider_request.model = selected_model

        # Update current provider and model
        self._current_provider = selected_provider
        self._current_model = selected_model

        return selected_provider, provider_request

    def _validate_provider_response(
        self, response: Any, request: ProviderLLMRequest
    ) -> bool:
        """Validate that a provider response is successful and meaningful.

        Args:
            response: The ProviderLLMResponse to validate
            request: The original ProviderLLMRequest that was sent

        Returns:
            True if response indicates successful generation, False otherwise
        """
        if not response:
            return False

        # Check basic response structure
        try:
            # Verify required fields are present and valid
            if not hasattr(response, "content") or not hasattr(response, "model"):
                return False

            # Check content is a string
            if not isinstance(response.content, str):
                return False

            # Content can be empty if there are tool calls
            has_tool_calls = hasattr(response, "tool_calls") and response.tool_calls
            if not response.content.strip() and not has_tool_calls:
                return False

            # Check model is specified and valid
            model = response.model
            if not isinstance(model, str) or not model.strip():
                return False

            # Verify model matches what was requested (if specified)
            if request.model and model != request.model:
                logger.warning(
                    f"Model mismatch: requested {request.model}, got {model}"
                )
                # This is not necessarily an error, but worth noting

            return True

        except (AttributeError, TypeError) as e:
            logger.error(f"Error validating response structure: {e}")
            return False

    def _record_successful_generation(self, provider_name: str, model: str) -> None:
        """Record that a provider successfully generated a response.

        This uses the provider's own health check mechanism and logs success
        for monitoring purposes, rather than directly manipulating health state.

        Args:
            provider_name: Name of the provider that succeeded
            model: Model that was used successfully
        """
        logger.debug(
            f"Provider {provider_name} successfully generated response using model {model}"
        )

        # Mark provider as successful through the public interface
        self.router.mark_provider_success(provider_name)

    def _record_failed_generation(self, provider_name: str, error: Exception) -> None:
        """Record that a provider failed to generate a response.

        Args:
            provider_name: Name of the provider that failed
            error: The exception that occurred
        """
        logger.error(f"Provider {provider_name} failed generation: {error}")

        # Mark provider as failed through the public interface
        self.router.mark_provider_failure(provider_name, error)

    async def _handle_generation_response(
        self,
        selected_provider: Any,
        provider_request: Any,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> LLMResponse:
        """Handle response generation with retry logic.

        Args:
            selected_provider: The selected provider
            provider_request: The provider request
            max_retries: Maximum number of retries
            retry_delay: Delay between retries

        Returns:
            The generated LLM response
        """
        try:
            # Generate response with retry logic
            provider_response = await selected_provider.generate_with_retry(
                provider_request, max_retries=max_retries, retry_delay=retry_delay
            )

            # Validate response before considering it successful
            if self._validate_provider_response(provider_response, provider_request):
                # Record successful generation
                self._record_successful_generation(
                    selected_provider.name, provider_response.model
                )
            else:
                # Response validation failed - this shouldn't happen if the provider is working correctly
                logger.warning(
                    f"Provider {selected_provider.name} returned invalid response format"
                )
                from local_coding_assistant.providers.exceptions import ProviderError

                self._record_failed_generation(
                    selected_provider.name, ProviderError("Invalid response format")
                )

            # Convert back to our response format
            return LLMResponse(
                content=provider_response.content,
                model_used=provider_response.model,
                tokens_used=provider_response.tokens_used,
                tool_calls=provider_response.tool_calls,
            )

        except Exception as e:
            # Record the failure
            self._record_failed_generation(selected_provider.name, e)
            # Re-raise the exception to be handled by the calling method
            raise

    async def _handle_fallback_generation(
        self,
        request: LLMRequest,
        provider: str | None = None,
        model: str | None = None,
        policy: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Handle fallback generation when primary provider fails.

        Args:
            request: The LLM request
            provider: Optional provider override
            model: Optional model override
            policy: Optional policy override
            overrides: Optional configuration overrides

        Returns:
            The generated response from fallback provider
        """
        # Extract model from overrides if not provided directly
        effective_model = model
        if effective_model is None and overrides:
            effective_model = overrides.get("llm.model_name")

        # Route to fallback provider
        selected_provider, provider_request = await self._route_request(
            request, provider, effective_model, policy, overrides, stream=False
        )

        # Generate response with fewer retries for fallback
        try:
            provider_response = await selected_provider.generate_with_retry(
                provider_request,
                max_retries=2,  # Fewer retries for fallback
                retry_delay=1.0,
            )

            # Validate fallback response as well
            if self._validate_provider_response(provider_response, provider_request):
                self._record_successful_generation(
                    selected_provider.name, provider_response.model
                )
            else:
                logger.warning(
                    f"Fallback provider {selected_provider.name} returned invalid response"
                )
                from local_coding_assistant.providers.exceptions import ProviderError

                self._record_failed_generation(
                    selected_provider.name, ProviderError("Invalid fallback response")
                )

            return LLMResponse(
                content=provider_response.content,
                model_used=provider_response.model,
                tokens_used=provider_response.tokens_used,
                tool_calls=provider_response.tool_calls,
            )

        except Exception as e:
            self._record_failed_generation(selected_provider.name, e)
            raise

    async def _handle_critical_error_generation(
        self,
        e: Exception,
        request: LLMRequest,
        provider: str | None = None,
        model: str | None = None,
        policy: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Handle critical errors with fallback logic for generation.

        Args:
            e: The original error
            request: The original LLM request
            provider: Optional provider override
            model: Optional model override
            policy: Optional policy override
            overrides: Optional configuration overrides

        Returns:
            The generated response from fallback provider if available
        """
        logger.error(f"Error during LLM generation: {e}")

        from local_coding_assistant.providers.exceptions import ProviderError

        # Check if this is a critical error that should trigger fallback
        if isinstance(e, ProviderError) and self.router.is_critical_error(e):
            # Mark the provider as unhealthy through public interface
            if hasattr(e, "provider") and e.provider:
                self.router.mark_provider_failure(e.provider, e)

            # Extract model from overrides if not provided directly
            effective_model = model
            if effective_model is None and overrides:
                effective_model = overrides.get("llm.model_name")

            # Try fallback generation
            try:
                return await self._handle_fallback_generation(
                    request, provider, effective_model, policy, overrides
                )
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")

        # If it's a provider exception, convert to LLMError
        if isinstance(e, ProviderError):
            raise LLMError(f"Provider error: {e}") from e
        else:
            raise LLMError(f"LLM generation failed: {e}") from e

    async def _handle_streaming_response(
        self,
        selected_provider: Any,
        provider_request: Any,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> AsyncIterator[str]:
        """Handle streaming response generation with retry logic.

        Args:
            selected_provider: The selected provider
            provider_request: The provider request
            max_retries: Maximum number of retries
            retry_delay: Delay between retries

        Yields:
            Content chunks from the streaming response
        """
        from local_coding_assistant.providers.exceptions import ProviderError

        total_content = ""
        has_valid_content = False

        try:
            async for delta in selected_provider.stream_with_retry(
                provider_request, max_retries=max_retries, retry_delay=retry_delay
            ):
                if delta.content:
                    total_content += delta.content
                    yield delta.content
                    has_valid_content = True

            # Validate the complete response before considering it successful
            if has_valid_content and total_content.strip():
                # Create a mock response object for validation
                mock_response = type(
                    "MockResponse",
                    (),
                    {"content": total_content, "model": provider_request.model},
                )()
                if self._validate_provider_response(mock_response, provider_request):
                    self._record_successful_generation(
                        selected_provider.name, provider_request.model
                    )
                else:
                    logger.warning(
                        f"Provider {selected_provider.name} streaming returned invalid content"
                    )
                    self._record_failed_generation(
                        selected_provider.name,
                        ProviderError("Invalid streaming response"),
                    )
            else:
                self._record_failed_generation(
                    selected_provider.name,
                    ProviderError("No valid content in streaming response"),
                )

        except Exception as e:
            # Record the failure
            self._record_failed_generation(selected_provider.name, e)
            # Re-raise the exception
            raise e

    async def _handle_fallback_streaming(
        self,
        request: LLMRequest,
        provider: str | None = None,
        model: str | None = None,
        policy: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Handle fallback streaming when primary provider fails.

        Args:
            request: The LLM request
            provider: Optional provider override
            model: Optional model override
            policy: Optional policy override
            overrides: Optional configuration overrides

        Yields:
            Content chunks from the fallback streaming response
        """
        from local_coding_assistant.providers.exceptions import ProviderError

        # Extract model from overrides if not provided directly
        effective_model = model
        if effective_model is None and overrides:
            effective_model = overrides.get("llm.model_name")

        # Route to fallback provider
        selected_provider, provider_request = await self._route_request(
            request, provider, effective_model, policy, overrides, stream=True
        )

        # Generate streaming response with fewer retries for fallback
        total_content = ""
        has_valid_content = False

        try:
            async for delta in selected_provider.stream_with_retry(
                provider_request,
                max_retries=2,  # Fewer retries for fallback
                retry_delay=1.0,
            ):
                if delta.content:
                    total_content += delta.content
                    yield delta.content
                    has_valid_content = True

            # Validate fallback streaming response as well
            if has_valid_content and total_content.strip():
                mock_response = type(
                    "MockResponse",
                    (),
                    {"content": total_content, "model": provider_request.model},
                )()
                if self._validate_provider_response(mock_response, provider_request):
                    self._record_successful_generation(
                        selected_provider.name, provider_request.model
                    )
                else:
                    logger.warning(
                        f"Fallback provider {selected_provider.name} streaming returned invalid content"
                    )
                    self._record_failed_generation(
                        selected_provider.name,
                        ProviderError("Invalid fallback streaming response"),
                    )
            else:
                self._record_failed_generation(
                    selected_provider.name,
                    ProviderError("No valid content in fallback streaming"),
                )

        except Exception as e:
            self._record_failed_generation(selected_provider.name, e)
            raise

    async def _handle_critical_error(
        self,
        e: Exception,
        request: LLMRequest,
        provider: str | None = None,
        model: str | None = None,
        policy: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Handle critical errors with fallback logic.

        Args:
            e: The original error
            request: The original LLM request
            provider: Optional provider override
            model: Optional model override
            policy: Optional policy override
            overrides: Optional configuration overrides

        Yields:
            Content chunks from fallback provider if available
        """
        logger.error(f"Error during LLM streaming generation: {e}")

        from local_coding_assistant.providers.exceptions import ProviderError

        # Check if this is a critical error that should trigger fallback
        if isinstance(e, ProviderError) and self.router.is_critical_error(e):
            # Mark the provider as unhealthy through public interface
            if hasattr(e, "provider") and e.provider:
                self.router.mark_provider_failure(e.provider, e)

            # Extract model from overrides if not provided directly
            effective_model = model
            if effective_model is None and overrides:
                effective_model = overrides.get("llm.model_name")

            # Try fallback streaming
            try:
                async for chunk in self._handle_fallback_streaming(
                    request, provider, effective_model, policy, overrides
                ):
                    yield chunk

            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                # If fallback streaming also fails, try fallback generation and yield as single chunk
                try:
                    fallback_response = await self._handle_fallback_generation(
                        request, provider, effective_model, policy, overrides
                    )
                    if fallback_response.content:
                        yield fallback_response.content
                except Exception as generation_fallback_error:
                    logger.error(
                        f"Generation fallback also failed: {generation_fallback_error}"
                    )

        else:
            # If it's not a critical error, re-raise the original error
            raise LLMError(f"LLM streaming generation failed: {e}") from e

    async def stream(
        self,
        request: LLMRequest,
        *,
        provider: str | None = None,
        model: str | None = None,
        policy: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Generate a streaming response using the provider system with fallback support.

        Yields:
            Content chunks from the streaming response as strings.
        """

        # Extract model from overrides if not provided directly
        effective_model = model
        if effective_model is None and overrides:
            effective_model = overrides.get("llm.model_name")

        try:
            # Route request and handle streaming response
            selected_provider, provider_request = await self._route_request(
                request, provider, effective_model, policy, overrides, stream=True
            )

            async for chunk in self._handle_streaming_response(
                selected_provider, provider_request
            ):
                yield chunk

        except Exception as e:
            # Handle critical errors with fallback logic

            async for chunk in self._handle_critical_error(
                e, request, provider, model, policy, overrides
            ):
                yield chunk

    def get_provider_status_list(self) -> list[dict[str, Any]]:
        """Get cached provider status list without triggering health checks.

        Returns:
            List of provider status dictionaries with name, source, status, and models
        """
        current_time = time.time()

        # Check if cache is still valid
        if current_time - self._last_health_check > self._cache_ttl:
            # Cache expired, refresh it synchronously if we're in an async context
            try:
                # Check if we're in an async context
                asyncio.get_running_loop()
                # We're in an async context, schedule the refresh
                task = asyncio.create_task(self._refresh_provider_status_cache())
                self._background_tasks.append(task)
                task.add_done_callback(self._background_tasks.remove)
            except RuntimeError:
                # No async context, refresh synchronously
                asyncio.run(self._refresh_provider_status_cache())

            # Get the current list of providers from the provider manager
        provider_names = self.provider_manager.list_providers()

        # Convert cached status to CLI format
        status_list = []
        for provider_name in provider_names:
            # Always get the current source directly from the provider manager
            source = (
                self.provider_manager.get_provider_source(provider_name) or "unknown"
            )

            # Get the cached status if it exists
            cached_status = self._provider_status_cache.get(provider_name, {})

            # Determine provider status
            if cached_status.get("status") == "not_implemented":
                status = "config"
                error = "Configured but no implementation"
                models = 0
            else:
                status = (
                    "available"
                    if cached_status.get("healthy", False)
                    else "unavailable"
                )
                error = cached_status.get("error")
                models = len(cached_status.get("models", []))

            # Add provider to status list
            status_list.append(
                {
                    "name": provider_name,
                    "source": source,
                    "status": status,
                    "models": models,
                    "error": error,
                }
            )

        return status_list

    def reload_providers(self):
        """Reload providers from configuration."""
        logger.info("Reloading providers through LLM manager")
        self.provider_manager.reload(self.config_manager)
        # Clear cache to force refresh on next access
        self._provider_status_cache.clear()
        self._last_health_check = 0

    async def get_provider_status(self) -> dict[str, dict[str, Any]]:
        """Get status information for all providers."""
        status = {}

        for provider_name in self.provider_manager.list_providers():
            provider = self.provider_manager.get_provider(provider_name)
            if provider:
                try:
                    health_status = await provider.health_check()

                    # Handle the case where health check is not available
                    if health_status == "unavailable":
                        status[provider_name] = {
                            "healthy": False,
                            "models": provider.get_available_models(),
                            "in_unhealthy_set": False,  # Not unhealthy, just not checkable
                            "status": "health_check_not_configured",
                        }
                    else:
                        is_healthy = bool(health_status)
                        status[provider_name] = {
                            "healthy": is_healthy,
                            "models": provider.get_available_models(),
                            "in_unhealthy_set": provider_name
                            in self.router.get_unhealthy_providers(),
                            "status": "healthy" if is_healthy else "unhealthy",
                        }
                except Exception as e:
                    logger.warning(f"Error checking health for {provider_name}: {e!s}")
                    status[provider_name] = {
                        "healthy": False,
                        "models": provider.get_available_models() if provider else [],
                        "error": str(e),
                        "in_unhealthy_set": provider_name
                        in self.router.get_unhealthy_providers(),
                        "status": "error",
                    }
            else:
                # Config-only provider
                status[provider_name] = {
                    "healthy": False,
                    "models": [],
                    "error": "Configured but no implementation",
                    "in_unhealthy_set": True,
                    "status": "not_implemented",
                }

        # Update cache
        self._provider_status_cache = status
        self._last_health_check = time.time()

        return status

    async def _refresh_provider_status_cache(self):
        """Refresh the provider status cache."""
        logger.debug("Refreshing provider status cache")
        self._provider_status_cache = await self.get_provider_status()
        self._last_health_check = time.time()
