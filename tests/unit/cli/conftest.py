"""Fixtures for CLI unit tests."""

from __future__ import annotations

import pytest
from rich.console import Console

from local_coding_assistant.runtime.events.reporting import RunReport


@pytest.fixture
def rich_console() -> Console:
    """Rich console that captures rendered output for golden comparison."""
    return Console(record=True, width=100, color_system=None)


@pytest.fixture
def sample_run_report() -> RunReport:
    """Sample RunReport with one frame and two actions for rendering tests."""
    return RunReport(
        run_id="run_123",
        session_id="session_123",
        mode="frame",
        status="success",
        final_answer="All done",
        models_used=["mock-model1", "mock-model2"],
        input_tokens_used=10,
        output_tokens_used=10,
        iterations=1,
        frames=[
            {
                "frame_id": "frame-1",
                "iteration": 1,
                "result": {
                    "status": "completed",
                    "total_input_tokens": 10,
                    "total_output_tokens": 10,
                    "total_latency_ms": 15.0,
                    "final_answer": "All done",
                    "actions": [
                        {
                            "kind": "llm_message",
                            "name": "generate",
                            "input": {"prompt": "Hello"},
                            "output": "Hello world",
                            "metadata": {
                                "input": {"prompt": "Hello"},
                                "output": "Hello world",
                                "reasoning": "Thinking step by step",
                            },
                        },
                        {
                            "kind": "tool_call",
                            "name": "calculator",
                            "input": {"expression": "1+1"},
                            "output": {"result": 2},
                        },
                    ],
                },
            }
        ],
    )
