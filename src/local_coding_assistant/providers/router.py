"""
Provider routing system for intelligent provider and model selection.
"""

from local_coding_assistant.providers import (
    BaseProvider,
    OptionalParameters,
    ProviderLLMRequest,
    ProviderManager,
)
from local_coding_assistant.providers.exceptions import (
    ProviderAuthError,
    ProviderConnectionError,
    ProviderNotFoundError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderValidationError,
)
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
        request: ProviderLLMRequest,
        role: str | None = None,
        provider: str | None = None,
    ) -> tuple[BaseProvider, str]:
        """Get the best provider and model for a request with fallback logic.

        This method routes to the appropriate resolution strategy based on the request
        and optional provider specification.

        Args:
            request: The provider request containing model and parameters
            role: Optional role for policy-based routing
            provider: Optional provider name to use for this request

        Returns:
            Tuple of (provider, model_name) that should handle the request

        Raises:
            ProviderNotFoundError: If no suitable provider/model can be found
        """
        # If provider is specified, try to use it first
        if provider and request.model:
            try:
                provider_instance = self.provider_manager.get_provider(provider)
                if provider_instance and provider_instance.supports_model(
                    request.model
                ):
                    provider_instance.validate_request(request)
                    return provider_instance, request.model
                else:
                    logger.debug(
                        f"Provider {provider} doesn't support model {request.model}"
                    )
            except Exception as e:
                logger.debug(f"Error using provider {provider}: {e}")
                # Fall through to model-based routing if provider lookup fails
                pass

        # If we have a model but no provider, try to find a provider that supports it
        if request.model and not provider:
            try:
                return await self._resolve_model_only(request.model, request)
            except ProviderNotFoundError as e:
                logger.debug(f"No provider found for model {request.model}: {e}")
                # Fall through to policy-based routing
            except Exception as e:
                logger.debug(f"Error resolving model {request.model}: {e}")
                # Fall through to policy-based routing

        # If we get here, either no provider/model was specified or the specified ones failed
        # Fall back to policy-based routing
        logger.info(f"Falling back to policy-based routing for {role} role")

        return await self._route_by_policy(role, request)

    async def _check_provider_health_and_fallback(
        self,
        provider_name: str,
        role: str | None,
        request: ProviderLLMRequest,
        fallback_message: str,
    ) -> tuple[BaseProvider, str] | None:
        """Check if provider is healthy and fallback if not.

        Args:
            provider_name: Name of the provider to check
            role: Agent role for fallback routing
            request: The provider request with model parameters to validate
            fallback_message: Message for logging

        Returns:
            Tuple of (provider, model) from fallback routing, or None if healthy
        """
        if provider_name in self._unhealthy_providers:
            logger.warning(fallback_message)
            return await self._route_by_policy_with_fallback(
                role, request, exclude_providers={provider_name}
            )

        # Validate the request against the provider
        provider = self.provider_manager.get_provider(provider_name)
        if provider and hasattr(request, "model") and request.model:
            try:
                provider.validate_request(request)
            except ProviderValidationError as e:
                raise ValueError(
                    f"Invalid parameters for model '{request.model}': {e!s}"
                ) from e

        return None  # Provider is healthy, continue with normal flow

    async def _resolve_provider_and_model(
        self,
        provider_name: str,
        model_name: str,
        role: str | None,
        request: ProviderLLMRequest,
    ) -> tuple[BaseProvider, str]:
        """Resolve provider and model when both are specified."""
        provider = self.provider_manager.get_provider(provider_name)
        if not provider:
            raise ProviderNotFoundError(f"Provider {provider_name} not found")

        # Create a copy of the request with the specified provider and model
        model_request = request.model_copy(
            update={"provider": provider_name, "model": model_name}
        )

        # Check health and validate parameters
        fallback_result = await self._check_provider_health_and_fallback(
            provider_name,
            role,
            model_request,
            f"Provider {provider_name} is marked unhealthy, trying fallback",
        )
        if fallback_result:
            return fallback_result

        # Check if provider supports the model
        if not provider.supports_model(model_name):
            # Try to find a provider that supports this model and parameters
            for fallback_provider_name in self.provider_manager.list_providers():
                if (
                    fallback_provider_name == provider_name
                    or fallback_provider_name in self._unhealthy_providers
                ):
                    continue

                fallback_provider = self.provider_manager.get_provider(
                    fallback_provider_name
                )
                if not fallback_provider or not fallback_provider.supports_model(
                    model_name
                ):
                    continue

                try:
                    # Create a new request with the fallback provider
                    fallback_request = model_request.model_copy(
                        update={"provider": fallback_provider_name}
                    )
                    fallback_provider.validate_request(fallback_request)

                    logger.info(
                        f"Model '{model_name}' not found in provider '{provider_name}', "
                        f"falling back to provider '{fallback_provider_name}'"
                    )
                    return fallback_provider, model_name
                except ProviderValidationError as e:
                    logger.debug(f"Skipping provider {fallback_provider_name}: {e}")
                    continue

            # If we get here, no suitable fallback was found
            available_models = provider.get_available_models()
            raise ProviderNotFoundError(
                f"Model '{model_name}' not found in provider '{provider_name}'. "
                f"Available models: {', '.join(available_models) if available_models else 'None'}"
            )

        # If we get here, the provider supports the model
        # The parameter validation is already done in _check_provider_health_and_fallback
        return provider, model_name

    async def _resolve_provider_only(
        self,
        provider_name: str,
        role: str | None,
        request: ProviderLLMRequest,
    ) -> tuple[BaseProvider, str]:
        """Resolve provider when only provider name is specified."""
        provider = self.provider_manager.get_provider(provider_name)
        if not provider:
            raise ProviderNotFoundError(f"Provider {provider_name} not found")

        # Check health and validate parameters
        fallback_result = await self._check_provider_health_and_fallback(
            provider_name,
            role,
            request,
            f"Provider {provider_name} is marked unhealthy, trying fallback",
        )
        if fallback_result:
            return fallback_result

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
            except ProviderValidationError as e:
                logger.debug(f"Skipping model {model_name} for {provider_name}: {e}")
                continue

        # If we get here, no model supports the requested parameters
        raise ProviderNotFoundError(
            f"No models available for provider {provider_name} support the requested parameters"
        )

    async def _resolve_model_only(
        self,
        model_name: str,
        request: ProviderLLMRequest,
    ) -> tuple[BaseProvider, str]:
        """Resolve model when only model name is specified.

        Args:
            model_name: Name of the model to find a provider for
            request: The original provider request with parameters to validate

        Returns:
            Tuple of (provider, model_name) that can handle the request

        Raises:
            ProviderNotFoundError: If no suitable provider is found for the model
        """
        # First find all providers that support this model
        providers = []
        for provider_name in self.provider_manager.list_providers():
            if provider_name in self._unhealthy_providers:
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
            except ProviderValidationError as e:
                logger.debug(
                    f"Skipping provider {provider.name} for model {model_name}: {e}"
                )
                continue

        # If we get here, no provider supports the requested parameters
        raise ProviderNotFoundError(
            f"No providers support model {model_name} with the requested parameters"
        )

    async def _route_by_policy_with_fallback(
        self,
        role: str | None,
        request: ProviderLLMRequest,
        exclude_models: set[str] | None = None,
        exclude_providers: set[str] | None = None,
    ) -> tuple[BaseProvider, str]:
        """Route based on agent policies with fallback support.

        Args:
            role: The agent role for policy selection
            request: The provider request
            exclude_models: Set of model names to exclude from consideration
            exclude_providers: Set of provider names to exclude from consideration

        Returns:
            Tuple of (provider, model_name) that should handle the request

        Raises:
            ProviderNotFoundError: If no suitable provider/model can be found
        """
        exclude_models = set(exclude_models or [])
        exclude_providers = set(exclude_providers or [])

        # Add unhealthy providers to exclusions
        exclude_providers.update(self._unhealthy_providers)

        logger.debug(
            f"Routing with exclusions - models: {exclude_models}, providers: {exclude_providers}"
        )

        # Get agent config
        resolved_config = self.config_manager.resolve(
            call_overrides=request.parameters.model_dump() if request.parameters else {}
        )
        agent_config = resolved_config.agent

        # Determine role (default to "general" if not specified)
        effective_role = role or "general"

        # Get policy for role
        policy = agent_config.get_policy_for_role(effective_role)

        # Try each provider/model in the policy, skipping excluded/unhealthy ones
        for item in policy:
            if item == "fallback:any":
                # Use any available provider/model as fallback
                try:
                    return await self._get_any_available_provider(
                        exclude_models=exclude_models,
                        exclude_providers=exclude_providers,
                        request=request,
                    )
                except ProviderNotFoundError:
                    continue  # Try next policy item

            # Parse provider:model format
            if ":" in item:
                provider_name, model_name = item.split(":", 1)

                # Skip if provider or model is excluded
                if provider_name in exclude_providers or model_name in exclude_models:
                    logger.debug(
                        f"Skipping excluded provider '{provider_name}' or model '{model_name}'"
                    )
                    continue

                provider = self.provider_manager.get_provider(provider_name)
                if provider:
                    try:
                        # Create a copy of the request with the current model
                        model_request = request.model_copy(update={"model": model_name})
                        provider.validate_request(model_request)
                        return provider, model_name
                    except ProviderValidationError as e:
                        logger.debug(
                            f"Skipping model {model_name} for {provider_name}: {e}"
                        )
                        continue

        # If no policy matches, try to find any available provider
        return await self._get_any_available_provider(
            exclude_models=exclude_models,
            exclude_providers=exclude_providers,
            request=request,
        )

    async def _get_any_available_provider(
        self,
        exclude_models: set[str] | None = None,
        exclude_providers: set[str] | None = None,
        request: ProviderLLMRequest | None = None,
    ) -> tuple[BaseProvider, str]:
        """Get any available provider and model that meets the criteria.

        Args:
            exclude_models: Set of model names to exclude
            exclude_providers: Set of provider names to exclude
            request: Optional request to validate against the model

        Returns:
            Tuple of (provider, model_name) that is available

        Raises:
            ProviderNotFoundError: If no available provider/model is found
        """
        exclude_models = set(exclude_models or [])
        exclude_providers = set(exclude_providers or [])

        # Create a minimal validation request if none provided
        validation_request = request or ProviderLLMRequest(
            messages=[{"role": "user", "content": "validation"}],
            model="",  # Will be set to the current model being checked
            parameters=OptionalParameters(),
        )

        # Try each provider in the manager
        for provider_name in self.provider_manager.list_providers():
            if (
                provider_name in exclude_providers
                or provider_name in self._unhealthy_providers
            ):
                logger.debug(f"Skipping excluded/unhealthy provider: {provider_name}")
                continue

            try:
                provider = self.provider_manager.get_provider(provider_name)
                if not provider:
                    continue

                # Find the first available model that's not excluded and passes validation
                for model_name in provider.get_available_models():
                    if model_name not in exclude_models:
                        try:
                            # Update the request with current model for validation
                            validation_request.model = model_name
                            provider.validate_request(validation_request)
                            logger.debug(
                                f"Validated provider {provider_name} with model {model_name}"
                            )
                            return provider, model_name
                        except ProviderValidationError as e:
                            logger.debug(
                                f"Model {model_name} from {provider_name} failed validation: {e}"
                            )
                            continue
            except Exception as e:
                logger.debug(f"Error processing provider {provider_name}: {e}")
                continue

        raise ProviderNotFoundError(
            "No available providers found. "
            f"Excluded providers: {exclude_providers}, "
            f"Excluded models: {exclude_models}"
        )

    def _get_policy_config(self, role: str | None) -> dict:
        """Get the policy configuration for the specified role.

        Args:
            role: The role to get the policy for. If None, returns the default policy.

        Returns:
            Dictionary containing the policy configuration
        """
        # Default policy configuration
        default_policy = {"models": ["google_gemini:gemini-2.5-flash", "fallback:any"]}

        # If no role is specified, return the default policy
        if not role:
            return default_policy

        # Try to get role-specific policy from config
        try:
            if hasattr(self.config_manager, "global_config"):
                models = self.config_manager.global_config.agents.get_policy_by_role(
                    role
                )
                if models:
                    return {"models": models}
        except Exception as e:
            logger.warning(f"Error getting policy for role '{role}': {e}")

        # Fall back to default policy
        return default_policy

    def _should_skip_provider(
        self, provider_name: str, exclude_providers: set[str]
    ) -> bool:
        """Check if a provider should be skipped based on exclusions and health status."""
        return (
            provider_name in exclude_providers
            or provider_name in self._unhealthy_providers
        )

    def _validate_provider_model(
        self, provider_name: str, model_name: str, request: ProviderLLMRequest
    ) -> tuple[BaseProvider, str] | None:
        """Validate if a provider and model can handle the request."""
        provider = self.provider_manager.get_provider(provider_name)
        if not provider:
            logger.debug(f"Provider not found: {provider_name}")
            return None

        if not provider.supports_model(model_name):
            logger.debug(
                f"Model {model_name} not supported by provider {provider_name}"
            )
            return None

        try:
            model_request = request.model_copy(
                update={"provider": provider_name, "model": model_name}
            )
            provider.validate_request(model_request)
            logger.debug(f"Routing to provider {provider_name} and model {model_name}")
            return provider, model_name
        except ProviderValidationError as e:
            logger.debug(f"Validation failed for {provider_name}/{model_name}: {e}")
            return None

    async def _handle_fallback_any(
        self,
        exclude_models: set[str],
        exclude_providers: set[str],
        request: ProviderLLMRequest,
    ) -> tuple[BaseProvider, str] | None:
        """Handle the 'fallback:any' special case in the policy."""
        try:
            return await self._get_any_available_provider(
                exclude_models=exclude_models,
                exclude_providers=exclude_providers,
                request=request,
            )
        except ProviderNotFoundError:
            return None

    async def _route_by_policy(
        self,
        role: str | None,
        request: ProviderLLMRequest,
        exclude_models: set[str] | None = None,
        exclude_providers: set[str] | None = None,
    ) -> tuple[BaseProvider, str]:
        """Route based on agent policies.

        Args:
            role: The role to use for policy selection
            request: The provider request
            exclude_models: Optional set of model names to exclude
            exclude_providers: Optional set of provider names to exclude

        Returns:
            Tuple of (provider, model_name) that should handle the request

        Raises:
            ProviderNotFoundError: If no suitable provider/model is found
        """
        policy_config = self._get_policy_config(role)
        exclude_models = set(exclude_models or [])
        exclude_providers = set(exclude_providers or [])

        if hasattr(request, "model") and request.model:
            exclude_models.add(request.model)

        # Process each model specification in the policy
        for model_spec in policy_config.get("models", []):
            if model_spec == "fallback:any":
                if result := await self._handle_fallback_any(
                    exclude_models, exclude_providers, request
                ):
                    return result
                continue

            # Parse provider:model specification
            parts = model_spec.split(":", 1)
            if len(parts) != 2:
                logger.warning(f"Invalid model specification in policy: {model_spec}")
                continue

            provider_name, model_name = parts

            # Skip if provider or model should be excluded
            if (
                self._should_skip_provider(provider_name, exclude_providers)
                or model_name in exclude_models
            ):
                logger.debug(f"Skipping provider: {provider_name}, model: {model_name}")
                continue

            # Try to validate and get the provider/model
            if result := self._validate_provider_model(
                provider_name, model_name, request
            ):
                return result

        # If we get here, no suitable provider/model was found in the policy
        # Try to find any available provider as a last resort
        try:
            return await self._get_any_available_provider(
                exclude_models=exclude_models,
                exclude_providers=exclude_providers,
                request=request,
            )
        except ProviderNotFoundError as e:
            raise ProviderNotFoundError(
                f"No available providers found that match the current policy. "
                f"Excluded providers: {exclude_providers}, "
                f"Excluded models: {exclude_models}"
            ) from e

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

    def get_unhealthy_providers(self) -> set[str]:
        """Get list of currently unhealthy providers."""
        return self._unhealthy_providers.copy()
