"""Provider resolution logic for finding appropriate providers and models."""

from local_coding_assistant.providers import (
    BaseProvider,
    ProviderLLMRequest,
    ProviderManager,
)
from local_coding_assistant.providers.exceptions import (
    ProviderNotFoundError,
    ProviderValidationError,
)
from local_coding_assistant.providers.health import ProviderHealthManager
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("providers.resolver")


class ProviderResolver:
    """Handles provider and model resolution with validation."""

    def __init__(
        self, provider_manager: ProviderManager, health_manager: ProviderHealthManager
    ):
        self.provider_manager = provider_manager
        self.health_manager = health_manager

    def _should_skip_provider(
        self, provider_name: str, exclude_providers: set[str] | None = None
    ) -> bool:
        """Check if a provider should be skipped."""
        exclude_providers = exclude_providers or set()
        return (
            provider_name in exclude_providers
            or not self.health_manager.is_provider_healthy(provider_name)
        )

    async def resolve_provider_and_model(
        self,
        provider_name: str,
        model_name: str,
        request: ProviderLLMRequest,
    ) -> tuple[BaseProvider, str]:
        """Resolve provider and model when both are specified."""
        provider = self.provider_manager.get_provider(provider_name)
        if not provider:
            raise ProviderNotFoundError(f"Provider {provider_name} not found")

        # Validate request against provider
        model_request = request.model_copy(
            update={"provider": provider_name, "model": model_name}
        )
        provider.validate_request(model_request)

        # Check if provider supports the model
        if not provider.supports_model(model_name):
            raise ProviderNotFoundError(
                f"Model '{model_name}' not found in provider '{provider_name}'"
            )

        return provider, model_name

    async def resolve_provider_only(
        self,
        provider_name: str,
        request: ProviderLLMRequest,
    ) -> tuple[BaseProvider, str]:
        """Resolve provider when only provider name is specified."""
        provider = self.provider_manager.get_provider(provider_name)
        if not provider:
            raise ProviderNotFoundError(f"Provider {provider_name} not found")

        # Get available models
        available_models = provider.get_available_models()
        if not available_models:
            raise ProviderNotFoundError(
                f"No models available for provider {provider_name}"
            )

        # Try to find a model that supports the requested parameters
        for model_name in available_models:
            try:
                # Create a copy of the request with the current model
                model_request = request.model_copy(
                    update={"provider": provider_name, "model": model_name}
                )
                provider.validate_request(model_request)
                return provider, model_name
            except ProviderValidationError:
                continue

        # If we get here, no model supports the requested parameters
        raise ProviderNotFoundError(
            f"No models available for provider {provider_name} support the requested parameters"
        )

    async def resolve_model_only(
        self,
        model_name: str,
        request: ProviderLLMRequest,
    ) -> tuple[BaseProvider, str]:
        """Resolve model when only model name is specified."""
        # Find all healthy providers that support this model
        providers = []
        for provider_name in self.provider_manager.list_providers():
            if self._should_skip_provider(provider_name):
                continue

            provider = self.provider_manager.get_provider(provider_name)
            if provider and provider.supports_model(model_name):
                providers.append(provider)

        if not providers:
            raise ProviderNotFoundError(
                f"No healthy providers found that support model {model_name}"
            )

        # Try to find a provider that supports the requested parameters
        for provider in providers:
            try:
                # Create a copy of the request with the current provider and model
                model_request = request.model_copy(
                    update={"provider": provider.name, "model": model_name}
                )
                provider.validate_request(model_request)
                logger.info(f"Found provider {provider.name} for model {model_name}")
                return provider, model_name
            except ProviderValidationError:
                continue

        # If we get here, no provider supports the requested parameters
        raise ProviderNotFoundError(
            f"No providers support model {model_name} with the requested parameters"
        )

    async def find_any_available_provider(
        self,
        request: ProviderLLMRequest,
    ) -> tuple[BaseProvider, str]:
        """Get any available provider and model that meets the criteria."""
        # Try each provider in the manager
        for provider_name in self.provider_manager.list_providers():
            if self._should_skip_provider(provider_name):
                continue

            try:
                provider = self.provider_manager.get_provider(provider_name)
                if not provider:
                    continue

                # Find the first available model that passes validation
                for model_name in provider.get_available_models():
                    try:
                        # Update the request with current model for validation
                        validation_request = request.model_copy(
                            update={"model": model_name}
                        )
                        provider.validate_request(validation_request)
                        return provider, model_name
                    except ProviderValidationError:
                        continue
            except Exception as e:
                logger.warning(f"Error processing provider {provider_name}: {e}")
                continue

        raise ProviderNotFoundError("No available providers found.")
