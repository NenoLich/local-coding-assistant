"""
Provider routing system for intelligent provider and model selection.
"""

from typing import Any

from local_coding_assistant.providers.base import BaseProvider
from local_coding_assistant.providers.exceptions import (
    ProviderAuthError,
    ProviderConnectionError,
    ProviderNotFoundError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from local_coding_assistant.providers.provider_manager import ProviderManager
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("providers.router")


class ProviderRouter:
    """Routes requests to appropriate providers based on policies and availability."""

    def __init__(self, config_manager, provider_manager: ProviderManager):
        self.config_manager = config_manager
        self.provider_manager = provider_manager
        self._provider_cache: dict[str, str] = {}
        self._unhealthy_providers: set[str] = set()

    def _is_critical_error(self, error: Exception) -> bool:
        """Check if an error is critical and should trigger fallback."""
        return isinstance(
            error,
            ProviderConnectionError
            | ProviderAuthError
            | ProviderRateLimitError
            | ProviderTimeoutError,
        )

    def _mark_provider_healthy(self, provider_name: str) -> None:
        """Mark a provider as healthy."""
        self._unhealthy_providers.discard(provider_name)

        # Update global config if available
        if (
            hasattr(self.config_manager, "global_config")
            and self.config_manager.global_config
        ):
            llm_config = self.config_manager.global_config.llm
            if hasattr(llm_config, "mark_provider_healthy"):
                llm_config.mark_provider_healthy(provider_name)

        logger.info(f"Marked provider {provider_name} as healthy")

    def _mark_provider_unhealthy(self, provider_name: str) -> None:
        """Mark a provider as unhealthy and update configuration."""
        self._unhealthy_providers.add(provider_name)

        # Update global config if available
        if (
            hasattr(self.config_manager, "global_config")
            and self.config_manager.global_config
        ):
            llm_config = self.config_manager.global_config.llm
            if hasattr(llm_config, "mark_provider_unhealthy"):
                llm_config.mark_provider_unhealthy(provider_name)

        logger.warning(f"Marked provider {provider_name} as unhealthy")

    async def get_provider_for_request(
        self,
        provider_name: str | None = None,
        model_name: str | None = None,
        role: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> tuple[BaseProvider, str]:
        """Get the best provider and model for a request with fallback logic.

        This method uses a dispatcher pattern to route to the appropriate
        resolution strategy based on the provided parameters.
        """
        # Route based on provided parameters using helper methods
        if provider_name and model_name:
            return await self._resolve_provider_and_model(
                provider_name, model_name, role, overrides
            )

        if provider_name:
            return await self._resolve_provider_only(provider_name, role, overrides)

        if model_name:
            return await self._resolve_model_only(model_name, role, overrides)

        # No specific parameters provided, use policy-based routing
        return await self._route_by_policy_with_fallback(role, overrides)

    async def _check_provider_health_and_fallback(
        self,
        provider_name: str,
        role: str | None,
        overrides: dict[str, Any] | None,
        fallback_message: str,
    ) -> tuple[BaseProvider, str] | None:
        """Check if provider is healthy and fallback if not.

        Args:
            provider_name: Name of the provider to check
            role: Agent role for fallback routing
            overrides: Configuration overrides
            fallback_message: Message for logging

        Returns:
            Tuple of (provider, model) from fallback routing, or None if healthy
        """
        if provider_name in self._unhealthy_providers:
            logger.warning(fallback_message)
            return await self._route_by_policy_with_fallback(
                role, overrides, exclude_providers={provider_name}
            )
        return None  # Provider is healthy, continue with normal flow

    async def _resolve_provider_and_model(
        self,
        provider_name: str,
        model_name: str,
        role: str | None,
        overrides: dict[str, Any] | None,
    ) -> tuple[BaseProvider, str]:
        """Resolve provider and model when both are specified."""
        provider = self.provider_manager.get_provider(provider_name)
        if not provider:
            raise ProviderNotFoundError(f"Provider {provider_name} not found")

        # Check health and fallback if needed
        fallback_result = await self._check_provider_health_and_fallback(
            provider_name,
            role,
            overrides,
            f"Provider {provider_name} is marked unhealthy, trying fallback",
        )
        if fallback_result:
            return fallback_result

        # Check if provider supports the model
        if not provider.supports_model(model_name):
            # Try to find a provider that supports this model
            fallback_provider = self._find_provider_for_model(model_name)
            if (
                fallback_provider
                and fallback_provider.name not in self._unhealthy_providers
            ):
                return fallback_provider, model_name

        return provider, model_name

    async def _resolve_provider_only(
        self,
        provider_name: str,
        role: str | None,
        overrides: dict[str, Any] | None,
    ) -> tuple[BaseProvider, str]:
        """Resolve provider when only provider name is specified."""
        provider = self.provider_manager.get_provider(provider_name)
        if not provider:
            raise ProviderNotFoundError(f"Provider {provider_name} not found")

        # Check health and fallback if needed
        fallback_result = await self._check_provider_health_and_fallback(
            provider_name,
            role,
            overrides,
            f"Provider {provider_name} is marked unhealthy, trying fallback",
        )
        if fallback_result:
            return fallback_result

        # Use first available model for this provider
        models = provider.get_available_models()
        if not models:
            raise ProviderNotFoundError(
                f"No models available for provider {provider_name}"
            )
        return provider, models[0]

    async def _resolve_model_only(
        self,
        model_name: str,
        role: str | None,
        overrides: dict[str, Any] | None,
    ) -> tuple[BaseProvider, str]:
        """Resolve model when only model name is specified."""
        provider = self._find_provider_for_model(model_name)
        if not provider:
            raise ProviderNotFoundError(f"No provider found for model {model_name}")

        # Check health and fallback if needed
        fallback_result = await self._check_provider_health_and_fallback(
            provider.name,
            role,
            overrides,
            f"Provider {provider.name} for model {model_name} is marked unhealthy, trying fallback",
        )
        if fallback_result:
            return fallback_result

        return provider, model_name

    def _find_provider_for_model(self, model_name: str) -> BaseProvider | None:
        """Find a provider that supports the given model."""
        # Check cache first
        if model_name in self._provider_cache:
            provider_name = self._provider_cache[model_name]
            return self.provider_manager.get_provider(provider_name)

        # Search through all providers
        for provider_name in self.provider_manager.list_providers():
            provider = self.provider_manager.get_provider(provider_name)
            if provider and provider.supports_model(model_name):
                self._provider_cache[model_name] = provider_name
                return provider

        return None

    async def _route_by_policy_with_fallback(
        self,
        role: str | None,
        overrides: dict[str, Any] | None,
        exclude_providers: set[str] | None = None,
    ) -> tuple[BaseProvider, str]:
        """Route based on agent policies with fallback support."""
        exclude_providers = exclude_providers or set()

        # Get agent config
        resolved_config = self.config_manager.resolve(overrides=overrides or {})
        agent_config = resolved_config.agent

        # Determine role (default to "general" if not specified)
        effective_role = role or "general"

        # Get policy for role
        policy = agent_config.get_policy_for_role(effective_role)

        # Try each provider/model in the policy, skipping unhealthy ones
        for item in policy:
            if item == "fallback:any":
                # Use any available provider/model as fallback
                return await self._get_any_available_provider(exclude_providers)

            # Parse provider:model format
            if ":" in item:
                provider_name, model_name = item.split(":", 1)
                if provider_name in exclude_providers:
                    continue

                provider = self.provider_manager.get_provider(provider_name)
                if (
                    provider
                    and provider.name not in self._unhealthy_providers
                    and provider.supports_model(model_name)
                ):
                    return provider, model_name

        # If no policy matches, use fallback
        return await self._get_any_available_provider(exclude_providers)

    async def _route_by_policy(
        self, role: str | None, overrides: dict[str, Any] | None
    ) -> tuple[BaseProvider, str]:
        """Route based on agent policies."""
        return await self._route_by_policy_with_fallback(role, overrides)

    async def _get_any_available_provider(
        self, exclude_providers: set[str] | None = None
    ) -> tuple[BaseProvider, str]:
        """Get any available provider and model."""
        exclude_providers = exclude_providers or set()

        for provider_name in self.provider_manager.list_providers():
            if provider_name in exclude_providers:
                continue

            provider = self.provider_manager.get_provider(provider_name)
            if provider and provider.name not in self._unhealthy_providers:
                models = provider.get_available_models()
                if models:
                    return provider, models[0]

        raise ProviderNotFoundError("No available providers found")

    def get_unhealthy_providers(self) -> set[str]:
        """Get list of currently unhealthy providers."""
        return self._unhealthy_providers.copy()

    def is_critical_error(self, error: Exception) -> bool:
        """Check if an error is considered critical for provider fallback.

        This is a public interface for checking if an error should trigger
        fallback behavior without actually marking the provider as unhealthy.

        Args:
            error: The exception to check

        Returns:
            True if the error is critical and should trigger fallback
        """
        return self._is_critical_error(error)

    def clear_unhealthy_providers(self) -> None:
        """Clear all unhealthy provider markings."""
        self._unhealthy_providers.clear()
        logger.info("Cleared all unhealthy provider markings")

    def mark_provider_success(self, provider_name: str) -> None:
        """Mark a provider as successful after generating a valid response.

        This is the public interface for indicating that a provider
        successfully generated a response and should be considered healthy.

        Args:
            provider_name: Name of the provider that succeeded
        """
        if provider_name in self._unhealthy_providers:
            self._mark_provider_healthy(provider_name)

    def mark_provider_failure(self, provider_name: str, error: Exception) -> None:
        """Mark a provider as failed after encountering an error.

        This is the public interface for indicating that a provider
        failed to generate a response and should be considered unhealthy
        if the error is critical.

        Args:
            provider_name: Name of the provider that failed
            error: The exception that occurred
        """
        if self._is_critical_error(error):
            self._mark_provider_unhealthy(provider_name)
