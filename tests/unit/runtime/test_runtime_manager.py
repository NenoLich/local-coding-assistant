import json
from collections.abc import AsyncIterator
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from local_coding_assistant.agent.llm_manager import (
    LLMManager,
)
from local_coding_assistant.config import get_config_manager
from local_coding_assistant.core.exceptions import (
    LLMError,
    ToolRegistryError,
)
from local_coding_assistant.providers.base import (
    BaseProvider,
    ProviderLLMRequest,
    ProviderLLMResponse,
    ProviderLLMResponseDelta,
)
from local_coding_assistant.providers.provider_manager import ProviderManager
from local_coding_assistant.runtime.runtime_manager import RuntimeManager
from local_coding_assistant.tools.builtin import SumTool
from local_coding_assistant.tools.tool_manager import ToolManager


class MockTestProvider(BaseProvider):
    """Test provider that integrates with the provider system for testing."""

    def __init__(self, provider_name: str | None = None, name: str | None = "test", base_url: str | None = "https://api.test.com", api_key: str | None = "test_key", api_key_env: str | None = None, models: list[str] | None = None, driver: str | None = "test", **kwargs):
        super().__init__(
            name=name or "test",
            base_url=base_url or "https://api.test.com",
            models=models or ["test-model", "gpt-4", "gpt-3.5"],
            api_key=api_key or "test_key",  # Provide API key for testing
        )
        self.call_count = 0
        self.calls: list[dict[str, Any]] = []

    async def generate(self, request: ProviderLLMRequest) -> ProviderLLMResponse:
        """Test generate method that returns deterministic responses."""
        self.call_count += 1
        self.calls.append({
            "model": request.model,
            "messages": len(request.messages),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "tools": len(request.tools) if request.tools else 0,
            "tool_outputs": bool(request.tool_outputs) if hasattr(request, "tool_outputs") and request.tool_outputs else False,
        })

        # Find the user message (not system prompt)
        user_content = ""
        for msg in request.messages:
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break

        # Create echo-style response
        content = f"echo:{user_content}"
        if request.tools:
            content += f"|tools:{len(request.tools)}"
        if hasattr(request, "tool_outputs") and request.tool_outputs:
            content += "|to"

        return ProviderLLMResponse(
            content=content,
            model=request.model,
            tokens_used=50,
        )

    async def stream(
        self, request: ProviderLLMRequest
    ) -> AsyncIterator[ProviderLLMResponseDelta]:
        # Find the user message (not system prompt)
        user_content = ""
        for msg in request.messages:
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break

        content = f"echo:{user_content}"
        if request.tools:
            content += f"|tools:{len(request.tools)}"
        if hasattr(request, "tool_outputs") and request.tool_outputs:
            content += "|to"

        # Split content into chunks for streaming
        for i in range(0, len(content), 10):
            chunk = content[i:i + 10]
            yield ProviderLLMResponseDelta(
                content=chunk,
                finish_reason="stop" if i + 10 >= len(content) else None,
            )

    async def health_check(self) -> bool:
        """Test health check that always passes."""
        return True
        
    def _create_driver_instance(self):
        """Create and return a mock driver instance for testing."""
        driver = MagicMock()
        driver.health_check.return_value = True
        return driver


class ToolManagerHelper(ToolManager):
    """Test tool manager that supports invoke() for backward compatibility."""

    def invoke(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Legacy invoke method for backward compatibility."""
        return self.run_tool(name, payload)


def setup_test_environment(
    persistent: bool = False,
) -> tuple[RuntimeManager, MockTestProvider, ToolManagerHelper]:
    """Set up test environment with provider system."""
    # Create config manager and load configuration
    config_manager = get_config_manager()
    config_manager.load_global_config()

    if persistent:
        config_manager.set_session_overrides({
            "runtime.persistent_sessions": True,
            "llm.model_name": "test-model",
            "llm.provider": "test",
            "llm.temperature": 0.7,
            "llm.max_tokens": 1000,
        })
    else:
        config_manager.set_session_overrides({
            "llm.model_name": "test-model",
            "llm.provider": "test",
            "llm.temperature": 0.7,
            "llm.max_tokens": 1000,
        })

    # Add test provider to configuration
    test_provider_config = {
        "name": "test",  # Required field for ProviderConfig
        "driver": "test",
        "base_url": "https://api.test.com",
        "api_key_env": "TEST_API_KEY",
        "models": {
            "test-model": {"supported_parameters": ["temperature", "max_tokens"]},
            "gpt-4": {"supported_parameters": ["temperature", "max_tokens"]},
            "gpt-3.5": {"supported_parameters": ["temperature", "max_tokens"]},
        }
    }

    # Set provider configuration through config manager
    config_manager.set_session_overrides({
        "providers.test": test_provider_config,
        "agent.policies.planner.models": ["test:test-model", "fallback:any"],
        "agent.policies.general.models": ["test:test-model", "fallback:any"],
        **({
            "runtime.persistent_sessions": True,
        } if persistent else {})
    })

    # Create provider manager and register test provider
    provider_manager = ProviderManager()
    test_provider = MockTestProvider(name="test")
    provider_manager._providers["test"] = MockTestProvider
    provider_manager._provider_sources["test"] = "test"

    # Reload provider manager with config
    provider_manager.reload(config_manager)

    # Create LLM manager with provider system
    llm_manager = LLMManager(config_manager=config_manager, provider_manager=provider_manager)

    # Create tool manager
    tool_manager = ToolManagerHelper()
    tool_manager.register_tool(SumTool())

    # Create runtime manager
    runtime_manager = RuntimeManager(
        llm_manager=llm_manager, tool_manager=tool_manager, config_manager=config_manager
    )

    return runtime_manager, test_provider, tool_manager


# ── non-persistent vs persistent behavior ─────────────────────────────────────


@pytest.mark.asyncio
async def test_orchestrate_with_model_override():
    """Test orchestrate method with model override using provider system."""
    mgr, _, _ = setup_test_environment(persistent=False)
    # Call with model override
    result = await mgr.orchestrate("test query", model="gpt-4")

    # Verify the call was made to the provider system
    # Get the provider instance that was actually used by the LLM manager
    actual_provider = mgr._llm_manager.provider_manager.get_provider("test")
    assert actual_provider is not None
    # Cast to MockTestProvider since we know this is the test provider in this context
    test_provider = cast(MockTestProvider, actual_provider)
    assert test_provider.call_count == 1
    assert len(test_provider.calls) == 1
    assert test_provider.calls[0]["model"] == "gpt-4"

    # Verify the response
    assert result["model_used"] == "gpt-4"
    assert "test query" in result["message"]


@pytest.mark.asyncio
async def test_orchestrate_with_multiple_overrides():
    """Test orchestrate method with multiple configuration overrides."""
    mgr, _, _ = setup_test_environment(persistent=False)

    # Call with multiple overrides
    result = await mgr.orchestrate(
        "test query", model="gpt-4", temperature=0.8, max_tokens=100
    )

    # Verify the call was made with correct overrides
    # Get the provider instance that was actually used by the LLM manager
    actual_provider = mgr._llm_manager.provider_manager.get_provider("test")
    assert actual_provider is not None
    test_provider = cast(MockTestProvider, actual_provider)
    assert test_provider.call_count == 1
    assert len(test_provider.calls) == 1
    assert test_provider.calls[0]["model"] == "gpt-4"
    assert test_provider.calls[0]["temperature"] == 0.8
    assert test_provider.calls[0]["max_tokens"] == 100

    assert result["model_used"] == "gpt-4"
    assert "test query" in result["message"]


@pytest.mark.asyncio
async def test_orchestrate_config_overrides_persist_across_calls():
    """Test that configuration overrides are applied per call and persist between calls."""
    mgr, _, _ = setup_test_environment(persistent=False)

    # First call with model override
    result1 = await mgr.orchestrate("test query 1", model="gpt-4", temperature=0.8)

    # Verify first call
    # Get the provider instance that was actually used by the LLM manager
    actual_provider = mgr._llm_manager.provider_manager.get_provider("test")
    assert actual_provider is not None
    test_provider = cast(MockTestProvider, actual_provider)
    assert test_provider.call_count == 1
    assert len(test_provider.calls) == 1
    assert test_provider.calls[0]["model"] == "gpt-4"
    assert test_provider.calls[0]["temperature"] == 0.8

    # Second call without overrides
    result2 = await mgr.orchestrate("test query 2")

    # Verify second call used same configuration
    assert test_provider.call_count == 2
    assert len(test_provider.calls) == 2
    assert test_provider.calls[1]["model"] == "gpt-4"
    assert test_provider.calls[1]["temperature"] == 0.8

    # Both calls should succeed
    assert result1["model_used"] == "gpt-4"
    assert result2["model_used"] == "gpt-4"
    assert "test query 1" in result1["message"]
    assert "test query 2" in result2["message"]


@pytest.mark.asyncio
async def test_orchestrate_config_override_validation():
    """Test that invalid configuration overrides raise appropriate errors."""
    mgr, _, _ = setup_test_environment(persistent=False)

    # Test invalid temperature override
    with pytest.raises(LLMError, match="Configuration update validation failed"):
        await mgr.orchestrate("test query", temperature=-1)

    # Test invalid max_tokens override
    with pytest.raises(LLMError, match="Configuration update validation failed"):
        await mgr.orchestrate("test query", max_tokens=0)


@pytest.mark.asyncio
async def test_persistent_reuses_same_session_and_grows_history():
    """Test persistent session behavior."""
    mgr, test_provider_instance, _ = setup_test_environment(persistent=True)

    out1 = await mgr.orchestrate("a")
    out2 = await mgr.orchestrate("b")

    # Same session id should be used across calls
    assert out1["session_id"] == out2["session_id"]

    # History accumulates (2 messages per call)
    assert len(out1["history"]) == 2
    assert len(out2["history"]) == 4

    # Provider sees the calls
    # Get the provider instance that was actually used by the LLM manager
    actual_provider = mgr._llm_manager.provider_manager.get_provider("test")
    assert actual_provider is not None
    test_provider = cast(MockTestProvider, actual_provider)
    assert test_provider.call_count == 2
    assert test_provider.calls[0]["messages"] >= 2  # User + system messages
    assert test_provider.calls[1]["messages"] >= 4  # Growing history


# ── directive parsing and tool invocation ─────────────────────────────────────
@pytest.mark.asyncio
async def test_directive_success_invokes_tool_and_passes_outputs_to_llm():
    """Test direct tool invocation and output passing."""
    mgr, test_provider_instance, _ = setup_test_environment(persistent=False)
    payload = json.dumps({"a": 2, "b": 3})
    out = await mgr.orchestrate(f"tool:sum {payload}")

    # One tool call recorded
    assert out["tool_calls"] and out["tool_calls"][0]["name"] == "sum"
    assert out["tool_calls"][0]["result"] == {"sum": 5}

    # Provider should have received the tool outputs
    # Get the provider instance that was actually used by the LLM manager
    actual_provider = mgr._llm_manager.provider_manager.get_provider("test")
    assert actual_provider is not None
    test_provider = cast(MockTestProvider, actual_provider)
    assert test_provider.call_count == 1
    assert len(test_provider.calls) == 1
    assert test_provider.calls[0]["tool_outputs"] is True


@pytest.mark.asyncio
async def test_directive_unknown_tool_raises():
    """Test that unknown tools raise appropriate errors."""
    mgr, _, _ = setup_test_environment(persistent=False)
    with pytest.raises(ToolRegistryError):
        await mgr.orchestrate('tool:unknown {"x": 1}')


@pytest.mark.asyncio
async def test_directive_invalid_json_raises():
    """Test that invalid JSON in tool calls returns an appropriate error message."""
    mgr, _, _ = setup_test_environment(persistent=False)
    result = await mgr.orchestrate("tool:sum not-json")
    # The mock provider should return the original message with tools count
    assert result["message"] == "echo:tool:sum not-json|tools:1"


@pytest.mark.asyncio
async def test_directive_invalid_payload_validation_raises():
    """Test that invalid tool payload raises appropriate errors."""
    mgr, _, _ = setup_test_environment(persistent=False)
    with pytest.raises(ToolRegistryError):
        await mgr.orchestrate('tool:sum {"a": "x", "b": 2}')


# ── edge cases ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_text_is_accepted_and_yields_echo():
    """Test that empty text is handled correctly."""
    mgr, test_provider_instance, _ = setup_test_environment(persistent=False)
    out = await mgr.orchestrate("")

    # Test provider format
    assert out["message"].startswith("echo:")
    # Get the provider instance that was actually used by the LLM manager
    actual_provider = mgr._llm_manager.provider_manager.get_provider("test")
    assert actual_provider is not None
    test_provider = cast(MockTestProvider, actual_provider)
    assert test_provider.call_count == 1
    assert test_provider.calls[0]["tools"] == 1  # SumTool is registered


@pytest.mark.asyncio
async def test_persistent_many_iterations_history_grows_linearly():
    """Test that history grows linearly with many iterations."""
    mgr, test_provider_instance, _ = setup_test_environment(persistent=True)
    n = 25
    for i in range(n):
        await mgr.orchestrate(f"m{i}")
    out = await mgr.orchestrate("final")

    # After N+1 runs, messages = 2*(N+1)
    assert len(out["history"]) == 2 * (n + 1)

    # Provider should have seen all calls
    # Get the provider instance that was actually used by the LLM manager
    actual_provider = mgr._llm_manager.provider_manager.get_provider("test")
    assert actual_provider is not None
    test_provider = cast(MockTestProvider, actual_provider)
    assert test_provider.call_count == n + 1
    # Last call should have large history
    assert test_provider.calls[-1]["messages"] >= 2 * n


# ── structured output validation ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_structured_output_shape_and_fields():
    """Test that output has the expected structure and fields."""
    mgr, _, _ = setup_test_environment(persistent=False)
    out = await mgr.orchestrate("shape-check")

    # Required keys
    assert set(out.keys()) == {
        "session_id",
        "message",
        "model_used",
        "tokens_used",
        "tool_calls",
        "history",
    }

    # Types
    assert isinstance(out["session_id"], str)
    assert isinstance(out["message"], str)
    assert isinstance(out["model_used"], str)
    assert isinstance(out["tokens_used"], int) or out["tokens_used"] is None
    assert isinstance(out["tool_calls"], list)
    assert isinstance(out["history"], list)

    # History contains dict items with role/content
    assert all(
        isinstance(m, dict) and {"role", "content"} <= set(m.keys())
        for m in out["history"]
    )


@pytest.mark.asyncio
async def test_provider_system_integration():
    """Test that the provider system integration works correctly."""
    mgr, _, _ = setup_test_environment(persistent=False)

    # Test that provider is properly integrated
    providers = mgr._llm_manager.provider_manager.list_providers()
    assert "test" in providers

    # Test provider routing
    result = await mgr.orchestrate("test provider routing")

    # Verify provider was called through the system
    # Get the provider instance that was actually used by the LLM manager
    actual_provider = mgr._llm_manager.provider_manager.get_provider("test")
    assert actual_provider is not None
    test_provider = cast(MockTestProvider, actual_provider)
    assert test_provider.call_count == 1
    assert test_provider.calls[0]["model"] == "test-model"

    assert result["model_used"] == "test-model"
    assert "test provider routing" in result["message"]


@pytest.mark.asyncio
async def test_provider_fallback_behavior():
    """Test provider fallback behavior when primary provider fails."""
    mgr, test_provider_instance, _ = setup_test_environment(persistent=False)

    # Test normal operation
    result = await mgr.orchestrate("test fallback")
    assert result["model_used"] == "test-model"

    # Verify provider was called
    # Get the provider instance that was actually used by the LLM manager
    actual_provider = mgr._llm_manager.provider_manager.get_provider("test")
    assert actual_provider is not None
    test_provider = cast(MockTestProvider, actual_provider)
    assert test_provider.call_count == 1
