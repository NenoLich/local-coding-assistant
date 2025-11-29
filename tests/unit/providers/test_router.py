"""Unit tests for ProviderRouter."""

import pytest

from unittest.mock import AsyncMock, MagicMock

from local_coding_assistant.providers.base import OptionalParameters, ProviderLLMRequest
from local_coding_assistant.providers.exceptions import (
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from local_coding_assistant.providers.router import ProviderRouter


@pytest.fixture
def config_manager():
    manager = MagicMock()
    manager.global_config = MagicMock()
    manager.global_config.llm = MagicMock()
    return manager


@pytest.fixture
def provider_manager():
    return MagicMock()


@pytest.fixture
def provider_request():
    return ProviderLLMRequest(
        messages=[{"role": "user", "content": "hello"}],
        model="initial-model",
        parameters=OptionalParameters(),
    )


@pytest.mark.asyncio
async def test_get_provider_for_request_with_specific_provider_success(
    config_manager, provider_manager, provider_request
):
    provider = MagicMock()
    provider.name = "primary"
    provider.supports_model.return_value = True

    provider_manager.get_provider.return_value = provider

    router = ProviderRouter(config_manager, provider_manager)

    result_provider, result_model = await router.get_provider_for_request(
        provider_request, provider="primary"
    )

    assert result_provider is provider
    assert result_model == provider_request.model
    provider.validate_request.assert_called_once_with(provider_request)


@pytest.mark.asyncio
async def test_get_provider_for_request_falls_back_to_policy_when_provider_invalid(
    config_manager, provider_manager, provider_request
):
    invalid_provider = MagicMock()
    invalid_provider.supports_model.return_value = False

    provider_manager.get_provider.return_value = invalid_provider

    router = ProviderRouter(config_manager, provider_manager)

    fallback_provider = MagicMock()
    fallback_provider.name = "fallback"

    router._route_by_policy = AsyncMock(return_value=(fallback_provider, "fallback-model"))

    result_provider, result_model = await router.get_provider_for_request(
        provider_request, role="general", provider="primary"
    )

    router._route_by_policy.assert_awaited_once()
    assert result_provider is fallback_provider
    assert result_model == "fallback-model"
    invalid_provider.validate_request.assert_not_called()


@pytest.mark.asyncio
async def test_get_provider_for_request_resolves_model_only(
    config_manager, provider_manager, provider_request
):
    provider_manager.list_providers.return_value = []

    router = ProviderRouter(config_manager, provider_manager)
    resolved_provider = MagicMock()

    router._resolve_model_only = AsyncMock(
        return_value=(resolved_provider, "resolved-model")
    )

    provider_request.model = "desired-model"

    result_provider, result_model = await router.get_provider_for_request(
        provider_request
    )

    router._resolve_model_only.assert_awaited_once_with(
        "desired-model", provider_request
    )
    assert result_provider is resolved_provider
    assert result_model == "resolved-model"


def test_mark_provider_success_clears_unhealthy_state(config_manager, provider_manager):
    router = ProviderRouter(config_manager, provider_manager)
    router._unhealthy_providers.add("provider-a")

    router.mark_provider_success("provider-a")

    assert "provider-a" not in router.get_unhealthy_providers()
    config_manager.global_config.llm.mark_provider_healthy.assert_called_once_with(
        "provider-a"
    )


def test_mark_provider_failure_marks_unhealthy_on_critical_error(
    config_manager, provider_manager
):
    router = ProviderRouter(config_manager, provider_manager)

    router.mark_provider_failure("provider-b", ProviderTimeoutError("timeout"))

    assert "provider-b" in router.get_unhealthy_providers()
    config_manager.global_config.llm.mark_provider_unhealthy.assert_called_once_with(
        "provider-b"
    )


def test_mark_provider_failure_skips_non_critical_error(
    config_manager, provider_manager
):
    router = ProviderRouter(config_manager, provider_manager)

    router.mark_provider_failure("provider-c", ValueError("not critical"))

    assert "provider-c" not in router.get_unhealthy_providers()
    config_manager.global_config.llm.mark_provider_unhealthy.assert_not_called()


@pytest.mark.parametrize(
    "error, expected",
    [
        (ProviderConnectionError("conn"), True),
        (ProviderRateLimitError("rate"), True),
        (ValueError("other"), False),
    ],
)
def test_is_critical_error_mapping(config_manager, provider_manager, error, expected):
    router = ProviderRouter(config_manager, provider_manager)

    assert router.is_critical_error(error) is expected
