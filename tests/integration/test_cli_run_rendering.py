from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

from local_coding_assistant.cli.commands import run as run_cli
from local_coding_assistant.cli.main import app
from local_coding_assistant.runtime.events import EventType, ExecutionEvent
from local_coding_assistant.runtime.reporting import RunReport


async def _fake_events_generator(
    text,
    *,
    agent_mode=None,
    model=None,
    temperature=None,
    max_tokens=None,
    tool_call_mode=None,
    sandbox_session=None,
):
    report = RunReport(
        run_id="run_test",
        session_id="session_test",
        mode="regular",
        status="success",
        final_answer="Hello from report",
        message="Hello from report",
        models_used=["mock-model"],
        tokens_used=12,
    )
    yield ExecutionEvent(
        type=EventType.TURN_COMPLETE, session_id="session_test", data={"report": report}
    )


def test_run_query_json_format(cli_runner, monkeypatch):
    runtime = AsyncMock()
    runtime.orchestrate = _fake_events_generator

    def fake_bootstrap(**kwargs):
        return {"runtime": runtime}

    monkeypatch.setattr(run_cli, "bootstrap", fake_bootstrap)

    result = cli_runner.invoke(app, ["run", "query", "Hello", "--format", "json"])

    assert result.exit_code == 0
    assert "Running query" in result.output
    assert '"final_answer"' in result.output


def test_run_query_frame_format(cli_runner, monkeypatch):
    runtime = AsyncMock()
    runtime.orchestrate = _fake_events_generator

    def fake_bootstrap(**kwargs):
        return {"runtime": runtime}

    monkeypatch.setattr(run_cli, "bootstrap", fake_bootstrap)

    result = cli_runner.invoke(app, ["run", "query", "Hello", "--format", "frame"])

    assert result.exit_code == 0
    assert "Frames" in result.output


def test_run_query_trace_written(cli_runner, monkeypatch, tmp_path: Path):
    runtime = AsyncMock()
    runtime.orchestrate = _fake_events_generator

    def fake_bootstrap(**kwargs):
        return {"runtime": runtime}

    monkeypatch.setattr(run_cli, "bootstrap", fake_bootstrap)

    trace_path = tmp_path / "trace.json"
    result = cli_runner.invoke(
        app,
        [
            "run",
            "query",
            "Hello",
            "--format",
            "json",
            "--trace",
            str(trace_path),
        ],
    )

    assert result.exit_code == 0
    assert trace_path.exists()

    payload = json.loads(trace_path.read_text())
    assert payload["final_answer"] == "Hello from report"
