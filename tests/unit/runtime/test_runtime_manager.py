from __future__ import annotations

import json
from typing import Any

import pytest

from local_coding_assistant.agent.llm_manager import LLMConfig, LLMManager, LLMResponse
from local_coding_assistant.config.schemas import RuntimeConfig
from local_coding_assistant.core.exceptions import AgentError, ToolRegistryError
from local_coding_assistant.runtime.runtime_manager import RuntimeManager
from local_coding_assistant.tools.builtin import SumTool
from local_coding_assistant.tools.tool_manager import ToolManager


class FakeLLM(LLMManager):
    """Test double for LLMManager with deterministic echo format.

    Echo format: "echo:{last_query}{suffix}|tools:{tool_count}", where suffix
    is "|to" when tool_outputs is provided.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.config = LLMConfig(model_name="fake-model", provider="fake")

    async def generate(self, request) -> LLMResponse:
        """Mock generate method that returns deterministic responses."""
        self.calls.append(
            {
                "session_id": request.context.get("session_id")
                if request.context
                else None,
                "history_len": len(request.context.get("history", []))
                if request.context
                else 0,
                "tool_calls": len(request.context.get("tool_calls", []))
                if request.context
                else 0,
                "model": self.config.model_name,
                "tool_outputs": request.tool_outputs,
            }
        )
        tool_count = len(request.tools) if request.tools else 0
        suffix = "|to" if request.tool_outputs else ""
        content = f"echo:{request.prompt or ''}{suffix}|tools:{tool_count}"

        return LLMResponse(
            content=content,
            model_used=self.config.model_name,
            tokens_used=50,
            tool_calls=None,
        )


class ToolManagerHelper(ToolManager):
    """Test tool manager that supports invoke() for backward compatibility."""

    def invoke(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Legacy invoke method for backward compatibility."""
        return self.run_tool(name, payload)


def make_manager(
    persistent: bool = False,
) -> tuple[RuntimeManager, FakeLLM, ToolManagerHelper]:
    llm = FakeLLM()
    tools = ToolManagerHelper()
    # Register the sum tool for testing
    tools.register_tool(SumTool())
    config = RuntimeConfig(persistent_sessions=persistent)
    mgr = RuntimeManager(llm_manager=llm, tool_manager=tools, config=config)
    return mgr, llm, tools


# ── non-persistent vs persistent behavior ─────────────────────────────────────


@pytest.mark.asyncio
async def test_orchestrate_with_model_override():
    """Test orchestrate method with model override."""
    mgr, llm, _ = make_manager(persistent=False)

    # Call with model override
    result = await mgr.orchestrate("test query", model="gpt-4")

    # Verify the call was made
    assert len(llm.calls) == 1

    # Config should be updated after the call (not restored)
    assert mgr._llm_manager.config.model_name == "gpt-4"

    # Verify the response
    assert result["model_used"] == "gpt-4"  # Should use the overridden model
    assert "test query" in result["message"]


@pytest.mark.asyncio
async def test_orchestrate_with_multiple_overrides():
    """Test orchestrate method with multiple configuration overrides."""
    mgr, _llm, _ = make_manager(persistent=False)

    # Call with multiple overrides
    result = await mgr.orchestrate(
        "test query", model="gpt-4", temperature=0.8, max_tokens=100
    )

    # Config should be updated after the call (not restored)
    assert mgr._llm_manager.config.model_name == "gpt-4"
    assert mgr._llm_manager.config.temperature == 0.8

    assert result["model_used"] == "gpt-4"
    assert "test query" in result["message"]


@pytest.mark.asyncio
async def test_orchestrate_config_persists_across_calls():
    """Test that configuration overrides persist across multiple orchestrate calls."""
    mgr, _llm, _ = make_manager(persistent=False)

    # First call with model override
    result1 = await mgr.orchestrate("test query 1", model="gpt-4", temperature=0.8)

    # Config should be updated
    assert mgr._llm_manager.config.model_name == "gpt-4"
    assert mgr._llm_manager.config.temperature == 0.8

    # Second call without overrides should use the updated config
    result2 = await mgr.orchestrate("test query 2")

    # Config should remain updated
    assert mgr._llm_manager.config.model_name == "gpt-4"
    assert mgr._llm_manager.config.temperature == 0.8

    # Both calls should succeed
    assert result1["model_used"] == "gpt-4"
    assert result2["model_used"] == "gpt-4"  # Should still use the overridden model
    assert "test query 1" in result1["message"]
    assert "test query 2" in result2["message"]


@pytest.mark.asyncio
async def test_orchestrate_config_override_validation():
    """Test that invalid configuration overrides raise appropriate errors."""
    mgr, _llm, _ = make_manager(persistent=False)

    # Test invalid temperature override
    with pytest.raises(AgentError, match="Configuration update validation failed"):
        await mgr.orchestrate("test query", temperature=-1)

    # Test invalid max_tokens override
    with pytest.raises(AgentError, match="Configuration update validation failed"):
        await mgr.orchestrate("test query", max_tokens=0)


@pytest.mark.asyncio
async def test_persistent_reuses_same_session_and_grows_history():
    mgr, llm, _ = make_manager(persistent=True)

    out1 = await mgr.orchestrate("a")
    out2 = await mgr.orchestrate("b")

    # Same session id should be used across calls
    assert out1["session_id"] == out2["session_id"]

    # History accumulates (2 messages per call)
    assert len(out1["history"]) == 2
    assert len(out2["history"]) == 4

    # LLM sees growing history
    assert llm.calls[0]["history_len"] == 0
    assert llm.calls[1]["history_len"] >= 2


# ── directive parsing and tool invocation ─────────────────────────────────────
@pytest.mark.asyncio
async def test_directive_success_invokes_tool_and_passes_outputs_to_llm():
    mgr, llm, _ = make_manager(persistent=False)
    payload = json.dumps({"a": 2, "b": 3})
    out = await mgr.orchestrate(f"tool:sum {payload}")

    # One tool call recorded
    assert out["tool_calls"] and out["tool_calls"][0]["name"] == "sum"
    assert out["tool_calls"][0]["result"] == {"sum": 5}

    # LLM got tool_outputs
    assert llm.calls[-1]["tool_outputs"] == {"sum": {"sum": 5}}


@pytest.mark.asyncio
async def test_directive_unknown_tool_raises():
    mgr, _, _ = make_manager(persistent=False)
    with pytest.raises(ToolRegistryError):
        await mgr.orchestrate('tool:unknown {"x": 1}')


@pytest.mark.asyncio
async def test_directive_invalid_json_raises():
    mgr, _, _ = make_manager(persistent=False)
    with pytest.raises(json.JSONDecodeError):
        await mgr.orchestrate("tool:sum not-json")


@pytest.mark.asyncio
async def test_directive_invalid_payload_validation_raises():
    mgr, _, _ = make_manager(persistent=False)
    with pytest.raises(ToolRegistryError):
        await mgr.orchestrate('tool:sum {"a": "x", "b": 2}')


# ── edge cases ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_text_is_accepted_and_yields_echo():
    mgr, _, _ = make_manager(persistent=False)
    out = await mgr.orchestrate("")
    # FakeLLM format
    assert out["message"].startswith("echo:")
    assert out["message"].endswith("|tools:1")


@pytest.mark.asyncio
async def test_persistent_many_iterations_history_grows_linearly():
    mgr, llm, _ = make_manager(persistent=True)
    n = 25
    for i in range(n):
        await mgr.orchestrate(f"m{i}")
    out = await mgr.orchestrate("final")
    # After N+1 runs, messages = 2*(N+1)
    assert len(out["history"]) == 2 * (n + 1)
    # LLM saw large history length by the end
    assert llm.calls[-1]["history_len"] >= 2 * n


# ── structured output validation ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_structured_output_shape_and_fields():
    mgr, _, _ = make_manager(persistent=False)
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
