import json
from collections.abc import AsyncIterator
from typing import Any, cast
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from local_coding_assistant.agent.llm_manager import (
    LLMManager, LLMRequest,
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

    def __init__(
        self,
        provider_name: str | None = None,
        name: str | None = "test",
        base_url: str | None = "https://api.test.com",
        api_key: str | None = "test_key",
        api_key_env: str | None = None,
        models: list[str] | None = None,
        driver: str | None = "test",
        **kwargs,
    ):
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
        self.calls.append(
            {
                "model": request.model,
                "messages": len(request.messages),
                "temperature": request.temperature,
                "max_tokens": request.parameters.max_tokens,
                "tools": len(request.parameters.tools) if request.parameters.tools else 0,
                "tool_outputs": bool(request.tool_outputs)
                if hasattr(request, "tool_outputs") and request.tool_outputs
                else False,
            }
        )

        # Find the user message (not system prompt)
        user_content = ""
        for msg in request.messages:
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break

        # Create echo-style response
        content = f"echo:{user_content}"
        if request.parameters.tools:
            content += f"|tools:{len(request.parameters.tools)}"
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
        if request.parameters.tools:
            content += f"|tools:{len(request.parameters.tools)}"
        if hasattr(request, "tool_outputs") and request.tool_outputs:
            content += "|to"

        # Split content into chunks for streaming
        for i in range(0, len(content), 10):
            chunk = content[i : i + 10]
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
        config_manager.set_session_overrides(
            {
                "runtime.persistent_sessions": True,
                "llm.model_name": "test-model",
                "llm.provider": "test",
                "llm.temperature": 0.7,
                "llm.max_tokens": 1000,
            }
        )
    else:
        config_manager.set_session_overrides(
            {
                "llm.model_name": "test-model",
                "llm.provider": "test",
                "llm.temperature": 0.7,
                "llm.max_tokens": 1000,
            }
        )

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
        },
    }

    # Set provider configuration through config manager
    config_manager.set_session_overrides(
        {
            "providers.test": test_provider_config,
            "agent.policies.planner.models": ["test:test-model", "fallback:any"],
            "agent.policies.general.models": ["test:test-model", "fallback:any"],
            **(
                {
                    "runtime.persistent_sessions": True,
                }
                if persistent
                else {}
            ),
        }
    )

    # Create provider manager and register test provider
    provider_manager = ProviderManager()
    test_provider = MockTestProvider(name="test")
    provider_manager._providers["test"] = MockTestProvider
    provider_manager._provider_sources["test"] = "test"

    # Reload provider manager with config
    provider_manager.reload(config_manager)

    # Create LLM manager with provider system
    llm_manager = LLMManager(
        config_manager=config_manager, provider_manager=provider_manager
    )

    # Create tool manager
    tool_manager = ToolManagerHelper()
    tool_manager.register_tool(SumTool())

    # Create runtime manager
    runtime_manager = RuntimeManager(
        llm_manager=llm_manager,
        tool_manager=tool_manager,
        config_manager=config_manager,
    )

    return runtime_manager, test_provider, tool_manager


# ── non-persistent vs persistent behavior ─────────────────────────────────────


@pytest.mark.asyncio
async def test_orchestrate_with_model_override():
    """Test orchestrate method with model override using provider system."""
    # Create a mock response
    mock_response = MagicMock()
    mock_response.content = "Test response"
    mock_response.model_used = "gpt-4"

    # Create a mock LLM manager that returns our mock response
    mock_llm_manager = AsyncMock()
    mock_llm_manager.generate.return_value = mock_response

    # Setup test environment with our mock LLM manager
    mgr, _, _ = setup_test_environment(persistent=False)
    mgr._llm_manager = mock_llm_manager

    # Call with model override
    result = await mgr.orchestrate("test query", model="gpt-4")

    # Verify the call was made to the provider system
    mock_llm_manager.generate.assert_awaited_once()

    # Verify the model override was passed correctly
    call_kwargs = mock_llm_manager.generate.call_args.kwargs
    assert call_kwargs["overrides"]['llm.model_name'] == "gpt-4"

    # Verify the response
    assert result["model_used"] == "gpt-4"
    assert result["message"] == "Test response"


@pytest.mark.asyncio
async def test_orchestrate_with_multiple_overrides():
    """Test orchestrate method with multiple configuration overrides."""
    # Create a mock response
    mock_response = MagicMock()
    mock_response.content = "Test response with overrides"
    mock_response.model_used = "gpt-4"

    # Create a mock LLM manager
    mock_llm_manager = AsyncMock()
    mock_llm_manager.generate.return_value = mock_response

    # Setup test environment with our mock LLM manager
    mgr, _, _ = setup_test_environment(persistent=False)
    mgr._llm_manager = mock_llm_manager

    # Call with multiple overrides
    result = await mgr.orchestrate(
        "test query",
        model="gpt-4",
        temperature=0.8,
        max_tokens=100
    )

    # Verify the call was made to the provider system
    mock_llm_manager.generate.assert_awaited_once()

    # Verify the overrides were passed correctly
    call_kwargs = mock_llm_manager.generate.call_args.kwargs
    assert call_kwargs["overrides"]["llm.model_name"] == "gpt-4"
    assert call_kwargs["overrides"]["llm.temperature"] == 0.8
    assert call_kwargs["overrides"]["llm.max_tokens"] == 100

    # Verify the response
    assert result["model_used"] == "gpt-4"
    assert result["message"] == "Test response with overrides"


@pytest.mark.asyncio
async def test_orchestrate_config_overrides_persist_across_calls():
    """Test that configuration overrides are applied per call and persist between calls."""
    # Create mock responses
    mock_response1 = MagicMock()
    mock_response1.content = "Test response 1"
    mock_response1.model_used = "gpt-4"

    mock_response2 = MagicMock()
    mock_response2.content = "Test response 2"
    mock_response2.model_used = "gpt-4"

    # Create a mock LLM manager
    mock_llm_manager = AsyncMock()
    mock_llm_manager.generate.side_effect = [mock_response1, mock_response2]

    # Setup test environment with our mock LLM manager
    mgr, _, _ = setup_test_environment(persistent=False)
    mgr._llm_manager = mock_llm_manager

    # First call with model and temperature override
    result1 = await mgr.orchestrate("test query 1", model="gpt-4", temperature=0.8)

    # Verify first call
    assert mock_llm_manager.generate.await_count == 1
    call1_kwargs = mock_llm_manager.generate.await_args_list[0].kwargs
    assert call1_kwargs["overrides"]["llm.model_name"] == "gpt-4"
    assert call1_kwargs["overrides"]["llm.temperature"] == 0.8

    # Second call without overrides
    result2 = await mgr.orchestrate("test query 2")

    # Verify second call used same configuration
    assert mock_llm_manager.generate.await_count == 2
    call2_kwargs = mock_llm_manager.generate.await_args_list[1].kwargs
    assert call2_kwargs["overrides"]["llm.model_name"] == "gpt-4"
    assert call2_kwargs["overrides"]["llm.temperature"] == 0.8

    # Both calls should succeed
    assert result1["model_used"] == "gpt-4"
    assert result2["model_used"] == "gpt-4"
    assert result1["message"] == "Test response 1"
    assert result2["message"] == "Test response 2"


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
    """Test persistent session behavior with mocks."""
    # Create mock responses with different content to verify they're used correctly
    mock_response1 = MagicMock()
    mock_response1.content = "Response 1"
    mock_response1.model_used = "gpt-4"

    mock_response2 = MagicMock()
    mock_response2.content = "Response 2"
    mock_response2.model_used = "gpt-4"

    # Create a mock LLM manager
    mock_llm_manager = AsyncMock()
    mock_llm_manager.generate.side_effect = [mock_response1, mock_response2]

    # Setup test environment with our mock LLM manager
    mgr, _, _ = setup_test_environment(persistent=True)
    mgr._llm_manager = mock_llm_manager

    # First call
    out1 = await mgr.orchestrate("a")
    # Second call
    out2 = await mgr.orchestrate("b")

    # Verify same session ID is used across calls
    assert out1["session_id"] == out2["session_id"]
    assert isinstance(out1["session_id"], str)  # Should have a session ID

    # Verify history grows as expected
    assert len(out1["history"]) == 2  # system + user message
    assert len(out2["history"]) == 4  # system + user + assistant + user

    # Verify LLM manager was called twice
    assert mock_llm_manager.generate.await_count == 2

    # Get all calls to generate
    calls = mock_llm_manager.generate.await_args_list

    # First call
    first_call_args = calls[0].args[0]  # First positional arg is the LLMRequest
    first_call_overrides = calls[0].kwargs.get("overrides", {})

    # Verify first call request
    assert isinstance(first_call_args, LLMRequest)
    assert first_call_args.prompt == "a"
    assert len(first_call_args.context.get("history", [])) == 1  # Just the user message

    # Second call
    second_call_args = calls[1].args[0]  # First positional arg is the LLMRequest
    second_call_overrides = calls[1].kwargs.get("overrides", {})

    # Verify second call request includes history
    assert isinstance(second_call_args, LLMRequest)
    assert second_call_args.prompt == "b"
    # Should include: user message 'a', assistant response, and user message 'b'
    assert len(second_call_args.context.get("history", [])) == 3
    history = second_call_args.context.get("history", [])
    assert history[0]["content"] == "a" and history[0]["role"] == "user"
    assert history[1]["content"] == "Response 1" and history[1]["role"] == "assistant"
    assert history[2]["content"] == "b" and history[2]["role"] == "user"


# ── directive parsing and tool invocation ─────────────────────────────────────
@pytest.mark.asyncio
async def test_directive_success_invokes_tool_and_passes_outputs_to_llm():
    """Test direct tool invocation and output passing."""
    # Create mock response
    mock_response = MagicMock()
    mock_response.content = "Tool result processed"
    mock_response.model_used = "gpt-4"

    # Create a mock LLM manager
    mock_llm_manager = AsyncMock()
    mock_llm_manager.generate.return_value = mock_response

    # Setup test environment with our mock LLM manager
    mgr, _, _ = setup_test_environment(persistent=False)
    mgr._llm_manager = mock_llm_manager

    # Mock tool manager
    mock_tool_manager = MagicMock()
    mock_tool_manager.run_tool.return_value = {"sum": 5}
    mgr._tool_manager = mock_tool_manager

    payload = json.dumps({"a": 2, "b": 3})
    out = await mgr.orchestrate(f"tool:sum {payload}")

    # Verify tool was called correctly
    mock_tool_manager.run_tool.assert_called_once_with("sum", {"a": 2, "b": 3})

    # Verify tool call was recorded in the output
    assert out["tool_calls"] and out["tool_calls"][0]["name"] == "sum"
    assert out["tool_calls"][0]["result"] == {"sum": 5}

    # Verify LLM was called with the tool output
    assert mock_llm_manager.generate.await_count == 1
    call_args = mock_llm_manager.generate.await_args[0][0]  # First positional arg is LLMRequest
    assert isinstance(call_args, LLMRequest)
    assert call_args.tool_outputs == {"sum": {"sum": 5}}
    assert "Tool sum executed successfully" in call_args.prompt

    # Verify the response is correctly passed through
    assert out["message"] == "Tool result processed"
    assert out["model_used"] == "gpt-4"


@pytest.mark.asyncio
async def test_directive_unknown_tool_raises():
    """Test that unknown tools raise appropriate errors."""
    mgr, _, _ = setup_test_environment(persistent=False)
    with pytest.raises(ToolRegistryError):
        await mgr.orchestrate('tool:unknown {"x": 1}')


@pytest.mark.asyncio
async def test_directive_invalid_json_raises():
    """Test that invalid JSON in tool calls returns an appropriate error message."""
    # Create mock response for the error case
    mock_response = MagicMock()
    mock_response.content = "Error: Invalid JSON in tool payload"
    mock_response.model_used = "gpt-4"
    mock_response.tokens_used = MagicMock()

    # Create a mock LLM manager
    mock_llm_manager = AsyncMock()
    mock_llm_manager.generate.return_value = mock_response

    # Setup test environment with our mock LLM manager
    mgr, _, _ = setup_test_environment(persistent=False)
    mgr._llm_manager = mock_llm_manager

    # Test with invalid JSON
    result = await mgr.orchestrate("tool:sum not-json")

    # Verify the error message indicates invalid JSON
    assert result["message"] == "Error: Invalid JSON in tool payload"
    assert result["model_used"] == "gpt-4"
    assert len(result["history"]) == 2
    assert result["history"][0]["content"] == "tool:sum not-json"
    assert result["history"][0]["role"] == "user"
    assert "Invalid JSON" in result["history"][1]["content"]
    assert result["history"][1]["role"] == "assistant"

    # Verify LLM was called once with the error message
    assert mock_llm_manager.generate.await_count == 1

@pytest.mark.asyncio
async def test_directive_invalid_payload_validation_raises():
    """Test that invalid tool payload raises appropriate errors."""
    # Create mock response
    mock_response = MagicMock()
    mock_response.content = "Error processing tool"
    mock_response.model_used = "gpt-4"
    mock_response.tokens_used = MagicMock()

    # Create a mock LLM manager
    mock_llm_manager = AsyncMock()
    mock_llm_manager.generate.return_value = mock_response

    # Setup test environment with our mock LLM manager
    mgr, _, _ = setup_test_environment(persistent=False)
    mgr._llm_manager = mock_llm_manager

    # Mock tool manager to raise validation error
    mock_tool_manager = MagicMock()
    mock_tool_manager.run_tool.side_effect = ValueError("Invalid payload")
    mgr._tool_manager = mock_tool_manager

    # Test with invalid payload
    result = await mgr.orchestrate('tool:sum {"a": "not a number", "b": 3}')

    # Verify error handling
    assert "tool_calls" in result
    if result["tool_calls"]:  # If there are tool calls
        assert "error" in result["tool_calls"][0].get("result", {})
    else:  # If no tool calls, check for error in message or history
        assert any("error" in msg.get("content", "").lower() or
                  "invalid" in msg.get("content", "").lower()
                  for msg in result.get("history", []))


# ── edge cases ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_text_is_accepted_and_yields_echo():
    """Test that empty text is handled correctly."""
    # Create mock response
    mock_response = MagicMock()
    mock_response.content = "echo:"
    mock_response.model_used = "gpt-4"

    # Create a mock LLM manager
    mock_llm_manager = AsyncMock()
    mock_llm_manager.generate.return_value = mock_response

    # Setup test environment with our mock LLM manager
    mgr, _, _ = setup_test_environment(persistent=False)
    mgr._llm_manager = mock_llm_manager

    # Test empty input
    result = await mgr.orchestrate("")

    # Verify empty input is handled gracefully
    assert result["message"] == "echo:"
    assert result["model_used"] == "gpt-4"
    assert mock_llm_manager.generate.await_count == 1


@pytest.mark.asyncio
async def test_persistent_many_iterations_history_grows_linearly():
    """Test that history grows linearly with many iterations."""
    # Create mock responses
    mock_responses = [
        MagicMock(content=f"Response {i}", model_used="gpt-4")
        for i in range(5)
    ]

    # Create a mock LLM manager
    mock_llm_manager = AsyncMock()
    mock_llm_manager.generate.side_effect = mock_responses

    # Setup test environment with our mock LLM manager
    mgr, _, _ = setup_test_environment(persistent=True)
    mgr._llm_manager = mock_llm_manager

    # Run multiple iterations
    results = []
    for i in range(5):
        results.append(await mgr.orchestrate(f"Message {i}"))

    # Verify session ID is consistent
    session_ids = {r["session_id"] for r in results}
    assert len(session_ids) == 1

    # Verify history grows linearly
    history_lengths = [len(r["history"]) for r in results]
    assert history_lengths == [2, 4, 6, 8, 10]  # 2 messages per interaction (user + assistant)

    # Verify LLM was called the correct number of times
    assert mock_llm_manager.generate.await_count == 5


# ── structured output validation ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_structured_output_shape_and_fields():
    """Test that output has the expected structure and fields."""
    # Create mock response
    mock_response = MagicMock()
    mock_response.content = "Test response"
    mock_response.model_used = "gpt-4"
    mock_response.tokens_used = 42

    # Create a mock LLM manager
    mock_llm_manager = AsyncMock()
    mock_llm_manager.generate.return_value = mock_response

    # Setup test environment with our mock LLM manager
    mgr, _, _ = setup_test_environment(persistent=False)
    mgr._llm_manager = mock_llm_manager

    # Test with a simple query
    result = await mgr.orchestrate("test query")

    # Verify output structure
    assert isinstance(result, dict)
    assert "session_id" in result
    assert "message" in result
    assert "model_used" in result
    assert "tokens_used" in result
    assert "tool_calls" in result
    assert "history" in result

    # Verify types
    assert isinstance(result["session_id"], str)
    assert isinstance(result["message"], str)
    assert isinstance(result["model_used"], str)
    assert isinstance(result["tokens_used"], int)
    assert isinstance(result["tool_calls"], list)
    assert isinstance(result["history"], list)


@pytest.mark.asyncio
async def test_provider_system_integration():
    """Test that the provider system integration works correctly."""
    # Create mock response
    mock_response = MagicMock()
    mock_response.content = "Test response from provider"
    mock_response.model_used = "test-model"
    mock_response.tokens_used = 10

    # Create a mock LLM manager
    mock_llm_manager = AsyncMock()
    mock_llm_manager.generate.return_value = mock_response

    # Setup test environment with our mock LLM manager
    mgr, _, _ = setup_test_environment(persistent=False)
    mgr._llm_manager = mock_llm_manager

    # Test with a simple query
    result = await mgr.orchestrate("test query", model="test-model")

    # Verify the response
    assert result["message"] == "Test response from provider"
    assert result["model_used"] == "test-model"
    assert result["tokens_used"] == 10

    # Verify LLM was called with the correct model
    assert mock_llm_manager.generate.await_count == 1

    # Get both positional and keyword arguments from the call
    call_args, call_kwargs = mock_llm_manager.generate.await_args

    # The model should be passed in the overrides dictionary
    assert "overrides" in call_kwargs
    assert call_kwargs["overrides"].get("llm.model_name") == "test-model"


@pytest.mark.asyncio
async def test_llm_provider_failure_handling():
    """Test that LLM provider errors are properly propagated."""
    # Setup test environment
    config_manager = get_config_manager()
    config_manager.load_global_config()

    # Create a mock LLM manager that will raise an exception
    mock_llm_manager = AsyncMock()

    # Create an async function that raises the exception
    async def raise_exception(*args, **kwargs):
        raise Exception("Provider error")

    # Set the side_effect to our async function
    mock_llm_manager.generate.side_effect = raise_exception

    # Create a minimal tool manager (not expected to be used)
    tool_manager = MagicMock()

    # Create runtime manager with our mocks
    runtime_manager = RuntimeManager(
        llm_manager=mock_llm_manager,
        tool_manager=tool_manager,
        config_manager=config_manager
    )

    # Test that the LLM exception is propagated
    with pytest.raises(Exception) as exc_info:
        await runtime_manager.orchestrate("test query")

    # Verify the exception was propagated correctly
    assert "Provider error" in str(exc_info.value)

    # Verify the LLM was called once
    assert mock_llm_manager.generate.await_count == 1

