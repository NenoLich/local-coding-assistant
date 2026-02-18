from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

from local_coding_assistant.cli.main import app
from local_coding_assistant.cli.commands import run as run_cli
from local_coding_assistant.runtime.reporting import RunReport


def _fake_report() -> RunReport:
    return RunReport(
        run_id="run_test",
        session_id="session_test",
        mode="regular",
        status="success",
        final_answer="Hello from report",
        message="Hello from report",
        models_used=["mock-model"],
        tokens_used=12,
    )


def test_run_query_json_format(cli_runner, monkeypatch):
    runtime = AsyncMock()
    runtime.orchestrate.return_value = _fake_report()

    def fake_bootstrap(**kwargs):
        return {"runtime": runtime}

    monkeypatch.setattr(run_cli, "bootstrap", fake_bootstrap)

    result = cli_runner.invoke(app, ["run", "query", "Hello", "--format", "json"])

    assert result.exit_code == 0
    assert "{" in result.output
    assert '"final_answer"' in result.output


def test_run_query_frame_format(cli_runner, monkeypatch):
    runtime = AsyncMock()
    runtime.orchestrate.return_value = _fake_report()

    def fake_bootstrap(**kwargs):
        return {"runtime": runtime}

    monkeypatch.setattr(run_cli, "bootstrap", fake_bootstrap)

    result = cli_runner.invoke(app, ["run", "query", "Hello", "--format", "frame"])

    assert result.exit_code == 0
    assert "Frames" in result.output


def test_run_query_trace_written(cli_runner, monkeypatch, tmp_path: Path):
    runtime = AsyncMock()
    runtime.orchestrate.return_value = _fake_report()

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
