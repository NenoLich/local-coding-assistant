from __future__ import annotations

from typing import Any, Optional, List, Dict

import json
import pytest

from local_coding_assistant.runtime.runtime_manager import RuntimeManager
from local_coding_assistant.runtime.session import SessionState


class FakeLLM:
    """Test double for LLMManager with deterministic echo format.

    Echo format: "echo:{last_query}{suffix}|tools:{tool_count}", where suffix
    is "|to" when tool_outputs is provided.
    """

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def ask_with_context(
        self,
        session: SessionState,
        *,
        tools: Any,
        tool_outputs: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> str:
        self.calls.append(
            {
                "session_id": session.id,
                "history_len": len(session.history),
                "tool_calls": len(session.tool_calls),
                "model": model,
                "tool_outputs": tool_outputs,
            }
        )
        tool_count = len(list(tools)) if hasattr(tools, "__iter__") else 0
        suffix = "|to" if tool_outputs else ""
        return f"echo:{session.last_query or ''}{suffix}|tools:{tool_count}"


class DummyTools(list):
    """Minimal tool registry double supporting invoke()."""

    def invoke(self, name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if name == "sum":
            a, b = payload.get("a"), payload.get("b")
            if not isinstance(a, int) or not isinstance(b, int):
                raise ValueError("Invalid input for sum")
            return {"sum": a + b}
        raise ValueError(f"Unknown tool: {name}")


def make_manager(
    persistent: bool = False,
) -> tuple[RuntimeManager, FakeLLM, DummyTools]:
    llm = FakeLLM()
    tools = DummyTools()
    mgr = RuntimeManager(llm=llm, tools=tools, persistent=persistent)  # type: ignore[arg-type]
    return mgr, llm, tools


# ── non-persistent vs persistent behavior ─────────────────────────────────────


def test_non_persistent_creates_fresh_session_and_does_not_carry_history():
    mgr, llm, _ = make_manager(persistent=False)

    out1 = mgr.orchestrate("hello", model="m1")
    out2 = mgr.orchestrate("world", model="m2")

    # Different sessions each time
    assert out1["session_id"] != out2["session_id"]

    # Each history contains only the two messages (user+assistant)
    assert len(out1["history"]) == 2
    assert len(out2["history"]) == 2

    # LLM saw a fresh context each time (after user add => history_len == 1)
    assert llm.calls[0]["history_len"] == 1
    assert llm.calls[1]["history_len"] == 1


def test_persistent_reuses_same_session_and_grows_history():
    mgr, llm, _ = make_manager(persistent=True)

    out1 = mgr.orchestrate("a")
    out2 = mgr.orchestrate("b")

    # Same session id should be used across calls
    assert out1["session_id"] == out2["session_id"]

    # History accumulates (2 messages per call)
    assert len(out1["history"]) == 2
    assert len(out2["history"]) == 4

    # LLM sees growing history
    assert llm.calls[0]["history_len"] == 1
    assert llm.calls[1]["history_len"] >= 3


# ── directive parsing and tool invocation ─────────────────────────────────────


def test_directive_success_invokes_tool_and_passes_outputs_to_llm():
    mgr, llm, _ = make_manager(persistent=False)
    payload = json.dumps({"a": 2, "b": 3})
    out = mgr.orchestrate(f"tool:sum {payload}")

    # One tool call recorded
    assert out["tool_calls"] and out["tool_calls"][0]["name"] == "sum"
    assert out["tool_calls"][0]["result"] == {"sum": 5}

    # LLM got tool_outputs
    assert llm.calls[-1]["tool_outputs"] == {"sum": {"sum": 5}}


def test_directive_unknown_tool_raises():
    mgr, _, _ = make_manager(persistent=False)
    with pytest.raises(ValueError):
        mgr.orchestrate('tool:unknown {"x": 1}')


def test_directive_invalid_json_raises():
    mgr, _, _ = make_manager(persistent=False)
    with pytest.raises(json.JSONDecodeError):
        mgr.orchestrate("tool:sum not-json")


def test_directive_invalid_payload_validation_raises():
    mgr, _, _ = make_manager(persistent=False)
    with pytest.raises(ValueError):
        mgr.orchestrate('tool:sum {"a": "x", "b": 2}')


# ── edge cases ────────────────────────────────────────────────────────────────


def test_empty_text_is_accepted_and_yields_echo():
    mgr, _, _ = make_manager(persistent=False)
    out = mgr.orchestrate("")
    # FakeLLM format
    assert out["message"].startswith("echo:")
    assert out["message"].endswith("|tools:0")


def test_persistent_many_iterations_history_grows_linearly():
    mgr, llm, _ = make_manager(persistent=True)
    N = 25
    for i in range(N):
        mgr.orchestrate(f"m{i}")
    out = mgr.orchestrate("final")
    # After N+1 runs, messages = 2*(N+1)
    assert len(out["history"]) == 2 * (N + 1)
    # LLM saw large history length by the end
    assert llm.calls[-1]["history_len"] >= 2 * N + 1


# ── structured output validation ─────────────────────────────────────────────


def test_structured_output_shape_and_fields():
    mgr, _, _ = make_manager(persistent=False)
    out = mgr.orchestrate("shape-check")

    # Required keys
    assert set(out.keys()) == {"session_id", "message", "tool_calls", "history"}

    # Types
    assert isinstance(out["session_id"], str)
    assert isinstance(out["message"], str)
    assert isinstance(out["tool_calls"], list)
    assert isinstance(out["history"], list)

    # History contains dict items with role/content
    assert all(
        isinstance(m, dict) and {"role", "content"} <= set(m.keys())
        for m in out["history"]
    )
