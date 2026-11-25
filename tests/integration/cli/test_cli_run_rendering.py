"""Integration-style tests for ``run query`` output rendering."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from local_coding_assistant.cli.main import app
from local_coding_assistant.runtime.events.reporting import RunReport


class _MockAsyncContextManager:
    """Async context manager that yields a fixed context dict."""

    def __init__(self, ctx):
        self.ctx = ctx

    async def __aenter__(self):
        return self.ctx

    async def __aexit__(self, *args):
        return False


def _make_report() -> RunReport:
    return RunReport(
        run_id="run_test",
        session_id="session_test",
        mode="regular",
        status="success",
        final_answer="Hello from report",
        message="Hello from report",
        models_used=["mock-model"],
        input_tokens_used=12,
        output_tokens_used=8,
    )


def _invoke_render(cli_runner, query, report, fmt, trace_path=None):
    """Invoke ``run query`` with mocked bootstrap + execute_query."""
    mock_runtime = MagicMock()
    mock_runtime.event_pipeline = MagicMock()
    mock_ctx = {"runtime": mock_runtime}

    mock_bootstrap = MagicMock()
    mock_bootstrap.return_value = _MockAsyncContextManager(mock_ctx)

    mock_execute_query = AsyncMock(return_value=(report, MagicMock(id="sess-1")))

    args = ["run", "query", query, "--format", fmt]
    if trace_path is not None:
        args.extend(["--trace", str(trace_path)])

    with (
        patch("local_coding_assistant.cli.commands.run.bootstrap", mock_bootstrap),
        patch(
            "local_coding_assistant.cli.commands.run.execute_query",
            mock_execute_query,
        ),
    ):
        return cli_runner.invoke(app, args)


class TestRunRendering:
    """Integration-style tests for run-query rendering."""

    def test_run_query_json_format(self, cli_runner):
        """JSON rendering includes the final answer field."""
        result = _invoke_render(cli_runner, "Hello", _make_report(), "json")

        assert result.exit_code == 0, result.stderr
        assert '"final_answer"' in result.stdout
        assert "Hello from report" in result.stdout

    def test_run_query_frame_format(self, cli_runner):
        """Frame rendering includes the frames header."""
        result = _invoke_render(cli_runner, "Hello", _make_report(), "frame")

        assert result.exit_code == 0, result.stderr
        assert "Frames" in result.stdout

    def test_run_query_trace_written(self, cli_runner, tmp_path: Path):
        """A trace file is written when --trace is provided."""
        trace_path = tmp_path / "trace.json"
        result = _invoke_render(
            cli_runner, "Hello", _make_report(), "json", trace_path=trace_path
        )

        assert result.exit_code == 0, result.stderr
        assert trace_path.exists()

        payload = json.loads(trace_path.read_text())
        assert payload["final_answer"] == "Hello from report"
