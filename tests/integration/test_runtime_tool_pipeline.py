from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_runtime_orchestrate_invokes_sum_tool_and_echoes_with_tool_outputs(
    ctx_with_mocked_llm,
):
    """Test that RuntimeManager properly invokes tools and provides outputs to LLM."""
    runtime = ctx_with_mocked_llm.get("runtime")

    # Test tool invocation through runtime orchestration
    out = await runtime.orchestrate('tool:sum {"a": 3, "b": 4}')

    # Tool call should be recorded in the result
    assert out["tool_calls"], "Expected at least one tool call recorded"
    call = out["tool_calls"][0]
    assert call["name"] == "sum"
    assert call["result"] == {"sum": 7}

    # LLM should receive tool outputs in the next message
    # The mock LLM should echo that it received tool outputs
    assert "[LLMManager] Echo:" in out["message"]
    assert "with tool outputs" in out["message"]
