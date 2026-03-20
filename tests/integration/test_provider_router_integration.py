"""Integration tests for ProviderResolver and ProviderHealthManager."""

from __future__ import annotations

import pytest

from local_coding_assistant.providers.base import OptionalParameters, ProviderLLMRequest
from local_coding_assistant.providers.exceptions import ProviderValidationError
from local_coding_assistant.providers.health import ProviderHealthManager
from local_coding_assistant.providers.resolver import ProviderResolver


class DummyConfigManager:
    """Minimal config manager for health manager testing."""

    def __init__(self):
        self.global_config = None


class FakeProvider:
    """Simple provider implementation for resolver testing."""

    def __init__(self, name: str, supported_models: set[str]) -> None:
        self.name = name
        self._supported_models = supported_models
        self.validate_calls: list[str] = []

    def supports_model(self, model: str) -> bool:
        return model in self._supported_models

    def get_available_models(self) -> list[str]:
        return list(self._supported_models)

    def validate_request(self, request: ProviderLLMRequest) -> None:
        self.validate_calls.append(request.model)
        if request.model not in self._supported_models:
            raise ProviderValidationError(
                f"Model '{request.model}' is not supported",
                provider=self.name,
                model=request.model,
            )


class StubProviderManager:
    """Provider manager facade for resolver testing."""

    def __init__(self, providers: dict[str, FakeProvider]) -> None:
        self._providers = providers

    def get_provider(self, name: str) -> FakeProvider | None:
        return self._providers.get(name)

    def list_providers(self) -> list[str]:
        return list(self._providers.keys())


@pytest.mark.asyncio
async def test_provider_resolver_find_any_available_provider() -> None:
    """Test that resolver finds any available healthy provider and model."""
    providers = {
        "primary": FakeProvider("primary", {"model-x", "model-z"}),
        "backup": FakeProvider("backup", {"model-y"}),
    }
    provider_manager = StubProviderManager(providers)
    health_manager = ProviderHealthManager(DummyConfigManager())

    resolver = ProviderResolver(provider_manager, health_manager)

    request = ProviderLLMRequest(
        messages=[{"role": "user", "content": "test"}],
        model="any",  # Will be overridden during resolution
        parameters=OptionalParameters(),
    )

    selected_provider, selected_model = await resolver.find_any_available_provider(
        request
    )

    # Should find the first available provider and model
    assert selected_provider.name == "primary"
    assert selected_model in ["model-x", "model-z"]
    assert selected_provider.validate_calls[-1] == selected_model


@pytest.mark.asyncio
async def test_provider_resolver_resolve_model_only() -> None:
    """Test resolving a specific model across available providers."""
    providers = {
        "provider_a": FakeProvider("provider_a", {"gpt-4"}),
        "provider_b": FakeProvider("provider_b", {"gpt-4", "claude-3"}),
    }
    provider_manager = StubProviderManager(providers)
    health_manager = ProviderHealthManager(DummyConfigManager())

    resolver = ProviderResolver(provider_manager, health_manager)

    request = ProviderLLMRequest(
        messages=[{"role": "user", "content": "test"}],
        model="gpt-4",
        parameters=OptionalParameters(),
    )

    selected_provider, selected_model = await resolver.resolve_model_only(
        "gpt-4", request
    )

    # Should find the first provider that supports the model
    assert selected_provider.name in ["provider_a", "provider_b"]
    assert selected_model == "gpt-4"
    assert selected_provider.validate_calls[-1] == "gpt-4"


@pytest.mark.asyncio
async def test_provider_resolver_resolve_provider_only() -> None:
    """Test resolving from a specific provider by finding suitable model."""
    providers = {
        "my_provider": FakeProvider("my_provider", {"model-a", "model-b"}),
    }
    provider_manager = StubProviderManager(providers)
    health_manager = ProviderHealthManager(DummyConfigManager())

    resolver = ProviderResolver(provider_manager, health_manager)

    request = ProviderLLMRequest(
        messages=[{"role": "user", "content": "test"}],
        model="dummy-model",  # Must have at least 1 character
        parameters=OptionalParameters(),
    )

    selected_provider, selected_model = await resolver.resolve_provider_only(
        "my_provider", request
    )

    assert selected_provider.name == "my_provider"
    assert selected_model in ["model-a", "model-b"]
    assert selected_provider.validate_calls[-1] == selected_model


@pytest.mark.asyncio
async def test_provider_resolver_resolve_provider_and_model() -> None:
    """Test resolving specific provider and model combination."""
    providers = {
        "target_provider": FakeProvider("target_provider", {"target-model"}),
    }
    provider_manager = StubProviderManager(providers)
    health_manager = ProviderHealthManager(DummyConfigManager())

    resolver = ProviderResolver(provider_manager, health_manager)

    request = ProviderLLMRequest(
        messages=[{"role": "user", "content": "test"}],
        model="target-model",
        parameters=OptionalParameters(),
    )

    selected_provider, selected_model = await resolver.resolve_provider_and_model(
        "target_provider", "target-model", request
    )

    assert selected_provider.name == "target_provider"
    assert selected_model == "target-model"
    assert selected_provider.validate_calls[-1] == "target-model"


@pytest.mark.asyncio
async def test_provider_resolver_skips_unhealthy_providers() -> None:
    """Test that resolver skips unhealthy providers."""
    providers = {
        "healthy": FakeProvider("healthy", {"model-x"}),
        "unhealthy": FakeProvider("unhealthy", {"model-x"}),
    }
    provider_manager = StubProviderManager(providers)
    health_manager = ProviderHealthManager(DummyConfigManager())

    # Mark one provider as unhealthy
    health_manager.mark_provider_failure("unhealthy", Exception("test error"))

    resolver = ProviderResolver(provider_manager, health_manager)

    request = ProviderLLMRequest(
        messages=[{"role": "user", "content": "test"}],
        model="model-x",
        parameters=OptionalParameters(),
    )

    selected_provider, selected_model = await resolver.resolve_model_only(
        "model-x", request
    )

    # Should only find the healthy provider
    assert selected_provider.name == "healthy"
    assert selected_model == "model-x"
    assert selected_provider.validate_calls[-1] == "model-x"

    # Unhealthy provider should not have been called
    assert len(providers["unhealthy"].validate_calls) == 0


def test_provider_health_manager_marking() -> None:
    """Test health manager provider status tracking."""
    config_manager = DummyConfigManager()
    health_manager = ProviderHealthManager(config_manager)

    # Initially no unhealthy providers
    assert len(health_manager.get_unhealthy_providers()) == 0

    # Mark provider as failed
    from local_coding_assistant.providers.exceptions import ProviderTimeoutError

    health_manager.mark_provider_failure(
        "test_provider", ProviderTimeoutError("test error")
    )
    assert "test_provider" in health_manager.get_unhealthy_providers()

    # Mark provider as successful
    health_manager.mark_provider_success("test_provider")
    assert "test_provider" not in health_manager.get_unhealthy_providers()


def test_provider_health_manager_critical_errors() -> None:
    """Test health manager recognizes critical errors."""
    from local_coding_assistant.providers.exceptions import ProviderTimeoutError

    config_manager = DummyConfigManager()
    health_manager = ProviderHealthManager(config_manager)

    # Critical error should trigger unhealthy marking
    critical_error = ProviderTimeoutError("timeout")
    assert health_manager.is_critical_error(critical_error)

    # Non-critical error should not trigger unhealthy marking
    non_critical_error = ValueError("validation error")
    assert not health_manager.is_critical_error(non_critical_error)


@pytest.mark.asyncio
async def test_provider_resolver_validation_error_handling() -> None:
    """Test resolver handles validation errors gracefully."""
    providers = {
        "provider": FakeProvider(
            "provider", {"valid-model"}
        ),  # Only supports one model
    }
    provider_manager = StubProviderManager(providers)
    health_manager = ProviderHealthManager(DummyConfigManager())

    resolver = ProviderResolver(provider_manager, health_manager)

    request = ProviderLLMRequest(
        messages=[{"role": "user", "content": "test"}],
        model="invalid-model",  # Model not supported by provider
        parameters=OptionalParameters(),
    )

    # Should raise ProviderNotFoundError when no valid combination found
    with pytest.raises(Exception):  # ProviderNotFoundError
        await resolver.resolve_provider_and_model("provider", "invalid-model", request)
