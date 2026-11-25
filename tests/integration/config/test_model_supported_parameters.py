"""Integration tests for model supported_parameters from config/providers.default.yaml.

These tests verify that every parameter listed in a model's ``supported_parameters``
is a valid field of ``ProviderLLMRequest.parameters`` (``OptionalParameters``) and
survives ``validate_against_model`` in TRUNCATE mode. They also verify that
``extra_body`` declared in config is parsed into ``ModelConfig`` and forwarded
to the driver payload automatically.
"""

from pathlib import Path

import pytest
import yaml

from local_coding_assistant.config.schemas import ModelConfig
from local_coding_assistant.providers.base import (
    OptionalParameters,
    ProviderLLMRequest,
)
from local_coding_assistant.providers.compatible_drivers import (
    OpenAIChatCompletionsDriver,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "config" / "providers.default.yaml"


def _load_providers_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _iter_models(config):
    for provider_name, provider_cfg in config.get("providers", {}).items():
        for model_name, model_cfg in (provider_cfg.get("models") or {}).items():
            yield provider_name, model_name, model_cfg


def _normalize_param(param):
    """Map a supported_parameter name to its field location.

    Returns ('request', field) for top-level ProviderLLMRequest fields,
    ('parameters', field) for OptionalParameters fields, and
    ('parameters', 'extra_body') for dotted 'extra_body.*' entries.
    Returns None if the parameter is not a valid field anywhere.
    """
    request_fields = set(ProviderLLMRequest.model_fields.keys())
    optional_fields = set(OptionalParameters.model_fields.keys())

    if param in request_fields:
        return ("request", param)
    if param in optional_fields:
        return ("parameters", param)
    if param.startswith("extra_body."):
        if "extra_body" in optional_fields:
            return ("parameters", "extra_body")
        return None
    return None


class TestModelSupportedParametersFromConfig:
    """Validate supported_parameters declared in providers.default.yaml."""

    @pytest.fixture(scope="class")
    def providers_config(self):
        return _load_providers_config()

    def test_every_supported_parameter_is_a_valid_request_field(self, providers_config):
        """Each supported_parameter must be a field on ProviderLLMRequest or its parameters.

        Provider-specific pass-through params (not in the standard schema) are
        declared in config but not mapped to driver payload fields. They are
        collected here as a documented contract so new unmapped params surface.
        """
        provider_specific = {"repetition_penalty", "structured outputs"}
        unmapped = []

        for provider_name, model_name, model_cfg in _iter_models(providers_config):
            supported = model_cfg.get("supported_parameters") or []
            for param in supported:
                if _normalize_param(param) is None:
                    unmapped.append((provider_name, model_name, param))

        unmapped_params = {param for _, _, param in unmapped}
        assert unmapped_params <= provider_specific, (
            "New provider-specific supported_parameters detected (not present in "
            f"ProviderLLMRequest/OptionalParameters): {sorted(unmapped_params - provider_specific)}. "
            f"Known provider-specific: {sorted(provider_specific)}"
        )

    def test_supported_parameters_survive_truncate_validation(self, providers_config):
        """All declared supported_parameters must remain after TRUNCATE validation."""
        for provider_name, model_name, model_cfg in _iter_models(providers_config):
            supported = model_cfg.get("supported_parameters") or []
            if not supported:
                continue

            params = {}
            for field in supported:
                location = _normalize_param(field)
                if location is None or location[0] != "parameters":
                    continue
                params[field] = self._sample_value(field)
            request = ProviderLLMRequest(
                messages=[{"role": "user", "content": "test"}],
                model=model_name,
                parameters=OptionalParameters(**params),
            )

            before = set(request.parameters.model_fields_set)
            request.validate_against_model(
                supported, mode=ProviderLLMRequest.ValidationMode.TRUNCATE
            )
            after = set(request.parameters.model_fields_set)

            stripped = before - after
            assert not stripped, (
                f"{provider_name}/{model_name}: supported_parameters stripped by "
                f"TRUNCATE validation: {stripped}"
            )

    @staticmethod
    def _sample_value(field):
        """Return a valid sample value for an OptionalParameters field."""
        samples = {
            "tools": [{"type": "function", "function": {"name": "t"}}],
            "tool_choice": "auto",
            "stream": False,
            "response_format": {"type": "text"},
            "max_tokens": 100,
            "top_p": 0.9,
            "top_k": 40,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "seed": 42,
            "stop": ["\n"],
            "include_reasoning": True,
            "reasoning_effort": "low",
            "include_usage": True,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
        }
        return samples.get(field, "sample-value")


class TestExtraBodyConfigFlow:
    """Verify extra_body declared in config is parsed and forwarded to the driver."""

    def test_model_config_parses_extra_body_from_dict(self):
        cfg = ModelConfig.from_dict(
            "test-model",
            {
                "supported_parameters": ["max_tokens", "extra_body"],
                "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
            },
        )
        assert cfg.name == "test-model"
        assert cfg.extra_body == {"chat_template_kwargs": {"enable_thinking": True}}

    def test_model_config_extra_body_defaults_to_empty(self):
        cfg = ModelConfig.from_dict("test-model", {})
        assert cfg.extra_body == {}

    def test_driver_includes_extra_body_in_payload(self):
        """extra_body from request.parameters must appear in the litellm payload."""
        driver = OpenAIChatCompletionsDriver(
            api_key="test-key", base_url="https://example.com/v1"
        )
        request = ProviderLLMRequest(
            messages=[{"role": "user", "content": "Explain binary search."}],
            model="test-model",
            temperature=1.0,
            parameters=OptionalParameters(
                max_tokens=16384,
                top_p=0.95,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            ),
        )

        payload = driver._build_chat_payload(request, stream=False)

        assert payload.get("extra_body") == {
            "chat_template_kwargs": {"enable_thinking": True}
        }

    def test_extra_body_stripped_when_not_in_supported_parameters(self):
        """Without extra_body in supported_parameters, TRUNCATE removes it."""
        request = ProviderLLMRequest(
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
            parameters=OptionalParameters(
                max_tokens=100,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            ),
        )

        request.validate_against_model(
            ["max_tokens"], mode=ProviderLLMRequest.ValidationMode.TRUNCATE
        )

        assert (
            request.parameters.extra_body is None or request.parameters.extra_body == {}
        )

    def test_extra_body_preserved_when_in_supported_parameters(self):
        """With extra_body in supported_parameters, TRUNCATE keeps it."""
        request = ProviderLLMRequest(
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
            parameters=OptionalParameters(
                max_tokens=100,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            ),
        )

        request.validate_against_model(
            ["max_tokens", "extra_body"],
            mode=ProviderLLMRequest.ValidationMode.TRUNCATE,
        )

        assert request.parameters.extra_body == {
            "chat_template_kwargs": {"enable_thinking": True}
        }
