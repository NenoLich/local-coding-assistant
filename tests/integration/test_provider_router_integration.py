"""Integration tests for ProviderRouter coordinating with config and provider manager stubs."""

from __future__ import annotations

import types

import pytest

from local_coding_assistant.providers.base import OptionalParameters, ProviderLLMRequest
from local_coding_assistant.providers.exceptions import (
    ProviderTimeoutError,
    ProviderValidationError,
)
from local_coding_assistant.providers.router import ProviderRouter


class DummyLLMState:
    """Track healthy/unhealthy markings applied by the router."""

    def __init__(self) -> None:
        self.marked_healthy: list[str] = []
        self.marked_unhealthy: list[str] = []

    def mark_provider_healthy(self, name: str) -> None:
        self.marked_healthy.append(name)

    def mark_provider_unhealthy(self, name: str) -> None:
        self.marked_unhealthy.append(name)


class DummyAgentConfig:
    """Expose routing policies similar to the real AgentConfig object."""

    def __init__(self, policies: dict[str, list[str]]) -> None:
        self._policies = policies

    def get_policy_for_role(self, role: str) -> list[str]:
        if role in self._policies:
            return self._policies[role]
        if "general" in self._policies:
            return self._policies["general"]
        return []


class DummyGlobalConfig:
    """Container mimicking the structure accessed by ProviderRouter."""

    def __init__(self, policies: dict[str, list[str]]) -> None:
        self.llm = DummyLLMState()
        self.agent = DummyAgentConfig(policies)


class DummyConfigManager:
    """Minimal config manager implementation for integration scenarios."""

    def __init__(self, policies: dict[str, list[str]]) -> None:
        self._policies = policies
        self.global_config = DummyGlobalConfig(policies)

    def resolve(self, call_overrides: dict | None = None) -> DummyGlobalConfig:
        # In these tests overrides are not relevant; simply return the global config.
        return self.global_config

    def get_agent_config(self, role: str):
        models = self._policies.get(role, self._policies.get("general", []))
        return types.SimpleNamespace(llm_policy={"models": models})


class FakeProvider:
    """Simple provider implementation exposing the surface used by ProviderRouter."""

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
    """Provider manager facade used to back ProviderRouter in integration tests."""

    def __init__(self, providers: dict[str, FakeProvider]) -> None:
        self._providers = providers

    def get_provider(self, name: str) -> FakeProvider | None:
        return self._providers.get(name)

    def list_providers(self) -> list[str]:
        return list(self._providers.keys())


@pytest.mark.asyncio
async def test_policy_routing_selects_first_available_provider() -> None:
    policies = {"general": ["primary:model-x", "backup:model-y", "fallback:any"]}
    config_manager = DummyConfigManager(policies)

    providers = {
        "primary": FakeProvider("primary", {"model-x"}),
        "backup": FakeProvider("backup", {"model-y"}),
    }
    provider_manager = StubProviderManager(providers)

    router = ProviderRouter(config_manager, provider_manager)

    request = ProviderLLMRequest(
        messages=[{"role": "user", "content": "route"}],
        model="unknown",
        parameters=OptionalParameters(),
    )

    selected_provider, selected_model = await router.get_provider_for_request(
        request, role="general"
    )

    assert selected_provider is providers["primary"]
    assert selected_model == "model-x"
    assert providers["primary"].validate_calls[-1] == "model-x"


@pytest.mark.asyncio
async def test_resolve_provider_and_model_falls_back_when_unhealthy() -> None:
    policies = {"general": ["primary:model-x", "backup:model-x"]}
    config_manager = DummyConfigManager(policies)

    providers = {
        "primary": FakeProvider("primary", {"model-x"}),
        "backup": FakeProvider("backup", {"model-x"}),
    }
    provider_manager = StubProviderManager(providers)

    router = ProviderRouter(config_manager, provider_manager)

    # Mark the primary provider as unhealthy via the public failure API.
    router.mark_provider_failure("primary", ProviderTimeoutError("timeout"))
    assert "primary" in router.get_unhealthy_providers()
    assert config_manager.global_config.llm.marked_unhealthy == ["primary"]

    request = ProviderLLMRequest(
        messages=[{"role": "user", "content": "needs fallback"}],
        model="model-x",
        parameters=OptionalParameters(),
    )

    fallback_provider, model = await router._resolve_provider_and_model(
        "primary", "model-x", "general", request
    )

    assert fallback_provider is providers["backup"]
    assert model == "model-x"
    assert providers["backup"].validate_calls[-1] == "model-x"


@pytest.mark.asyncio
async def test_policy_fallback_any_uses_available_healthy_provider() -> None:
    policies = {"general": ["primary:model-x", "fallback:any"]}
    config_manager = DummyConfigManager(policies)

    providers = {
        "primary": FakeProvider("primary", {"model-x"}),
        "backup": FakeProvider("backup", {"model-y"}),
    }
    provider_manager = StubProviderManager(providers)

    router = ProviderRouter(config_manager, provider_manager)

    # Primary provider is unhealthy, so fallback:any should consider other providers.
    router.mark_provider_failure("primary", ProviderTimeoutError("timeout"))
    assert "primary" in router.get_unhealthy_providers()

    request = ProviderLLMRequest(
        messages=[{"role": "user", "content": "fallback"}],
        model="unknown",
        parameters=OptionalParameters(),
    )

    selected_provider, selected_model = await router.get_provider_for_request(
        request, role="general"
    )

    assert selected_provider is providers["backup"]
    assert selected_model == "model-y"
    assert providers["backup"].validate_calls[-1] == "model-y"

    # Ensure the unhealthy provider was not marked healthy during the process.
    assert config_manager.global_config.llm.marked_healthy == []
