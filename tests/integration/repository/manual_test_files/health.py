"""Provider health management for tracking healthy/unhealthy providers."""

from local_coding_assistant.providers.exceptions import (
    ProviderAuthError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("providers.health")


class ProviderHealthManager:
    """Manages provider health status and tracking."""

    def __init__(self, config_manager):
        self.config_manager = config_manager
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

    def is_critical_error(self, error: Exception) -> bool:
        """Check if an error is considered critical for provider fallback."""
        return self._is_critical_error(error)

    def clear_unhealthy_providers(self) -> None:
        """Clear all unhealthy provider markings."""
        self._unhealthy_providers.clear()
        logger.info("Cleared all unhealthy provider markings")

    def mark_provider_success(self, provider_name: str) -> None:
        """Mark a provider as successful after generating a valid response."""
        if provider_name in self._unhealthy_providers:
            self._mark_provider_healthy(provider_name)

    def mark_provider_failure(self, provider_name: str, error: Exception) -> None:
        """Mark a provider as failed after encountering an error."""
        if self._is_critical_error(error):
            self._mark_provider_unhealthy(provider_name)

    def get_unhealthy_providers(self) -> set[str]:
        """Get list of currently unhealthy providers."""
        return self._unhealthy_providers.copy()

    def is_provider_healthy(self, provider_name: str) -> bool:
        """Check if a provider is currently healthy."""
        return provider_name not in self._unhealthy_providers
