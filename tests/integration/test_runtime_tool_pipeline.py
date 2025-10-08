from __future__ import annotations

from local_coding_assistant.core.bootstrap import bootstrap


def test_runtime_orchestrate_invokes_sum_tool_and_echoes_with_tool_outputs():
    ctx = bootstrap()
    runtime = ctx["runtime"]

    out = runtime.orchestrate('tool:sum {"a": 3, "b": 4}')

    # Tool call recorded
    assert out["tool_calls"], "Expected at least one tool call recorded"
    call = out["tool_calls"][0]
    assert call["name"] == "sum"
    assert call["result"] == {"sum": 7}

    # LLM echo should indicate tool outputs were provided
    assert "[LLMManager] Echo:" in out["message"]
    assert "with tool outputs" in out["message"]
