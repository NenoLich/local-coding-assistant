from __future__ import annotations

from datetime import UTC, datetime

import pytest
from rich.console import Console

from local_coding_assistant.runtime.execution_types import ExecutionStatus
from local_coding_assistant.runtime.reporting import RunMetrics, RunReport


@pytest.fixture
def rich_console() -> Console:
    return Console(record=True, width=100, color_system=None)


@pytest.fixture
def sample_run_report() -> RunReport:
    frame = {
        "id": "frame_1",
        "session_id": "session_123",
        "iteration": 1,
        "prompt_context": [],
        "rendered_prompt": [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"}
        ],
        "model_response_raw": "LLM response",
        "actions": [
            {
                "id": "action_1",
                "kind": "llm_message",
                "name": "generate",
                "started_at": datetime(2026, 1, 1, tzinfo=UTC),
                "finished_at": datetime(2026, 1, 1, tzinfo=UTC),
                "llm_metrics": {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                    "reasoning_tokens": 0,
                    "latency_ms": 500.0,
                    "model": "mock-model"
                },
                "tool_calls": [],
                "tool_trace": None,
                "metadata": {
                    "input": {"prompt": "Hello"},
                    "output": "Hello world",
                    "reasoning": "Thinking step by step"
                },
            },
            {
                "id": "action_2",
                "kind": "tool_call",
                "name": "calculator",
                "started_at": datetime(2026, 1, 1, tzinfo=UTC),
                "finished_at": datetime(2026, 1, 1, tzinfo=UTC),
                "llm_metrics": None,
                "tool_calls": [],
                "tool_trace": {
                    "call_id": "call_123",
                    "tool_name": "calculator",
                    "input": {"expression": "1+1"},
                    "output": {"result": 2},
                    "success": True,
                    "error": None,
                    "duration_ms": 12.0,
                    "start_time": datetime(2026, 1, 1, tzinfo=UTC),
                    "end_time": datetime(2026, 1, 1, tzinfo=UTC),
                    "parent_call_id": None,
                    "child_call_ids": [],
                    "execution_mode": "sync",
                    "source": "test",
                    "resource_metrics": [],
                    "metadata": {"success": True}
                },
                "metadata": {},
            },
        ],
        "result": {
            "status": ExecutionStatus.SUCCESS,
            "final_answer": "All done",
            "finish_reason": None,
            "total_latency_ms": 15.0,
            "total_tokens": 20,
            "error_message": None,
            "handler_context": None,
            "files_created": [],
            "files_modified": [],
        },
        "started_at": datetime(2026, 1, 1, tzinfo=UTC),
        "finished_at": datetime(2026, 1, 1, tzinfo=UTC),
    }

    return RunReport(
        run_id="run_123",
        session_id="session_123",
        mode="frame",
        status="success",
        final_answer="All done",
        message="All done",
        models_used=["mock-model1", "mock-model2"],
        tokens_used=20,
        iterations=1,
        frames=[frame],
        metrics=RunMetrics(tokens_used=20, total_latency_ms=15.0, tool_calls=1),
    )
