"""
End-to-end tests for run-query output (formerly streaming API compatibility).

These tests mock the bootstrap / execute-query boundary and verify that the
rendered report content reaches standard output for various configurations.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from local_coding_assistant.cli.main import app


class _MockAsyncContextManager:
    """Async context manager that yields a fixed context dict."""

    def __init__(self, ctx):
        self.ctx = ctx

    async def __aenter__(self):
        return self.ctx

    async def __aexit__(self, *args):
        return False


def _invoke_run(cli_runner, query, final_answer, extra_args=None):
    """Invoke ``run query`` with a mocked execute_query returning a report."""
    mock_runtime = MagicMock()
    mock_runtime.event_pipeline = MagicMock()
    mock_ctx = {"runtime": mock_runtime}

    mock_bootstrap = MagicMock()
    mock_bootstrap.return_value = _MockAsyncContextManager(mock_ctx)

    mock_report = MagicMock()
    mock_report.final_answer = final_answer
    mock_report.status = "success"

    mock_execute_query = AsyncMock(return_value=(mock_report, MagicMock(id="sess-1")))

    with (
        patch("local_coding_assistant.cli.commands.run.bootstrap", mock_bootstrap),
        patch(
            "local_coding_assistant.cli.commands.run.execute_query",
            mock_execute_query,
        ),
    ):
        args = ["run", "query", query]
        if extra_args:
            args.extend(extra_args)
        return cli_runner.invoke(app, args)


class TestStreamingAPICompatibilityE2E:
    """End-to-end tests for run-query output across configurations."""

    def test_event_buffering_into_traditional_results(self, cli_runner):
        """Buffered streaming results are rendered to standard output."""
        result = _invoke_run(
            cli_runner,
            "Test event buffering",
            "Buffered result: This is buffered content for compatibility. Calculator result: 4",
        )

        assert result.exit_code == 0, result.stderr
        assert "Buffered result:" in result.stdout
        assert "buffered content for compatibility" in result.stdout
        assert "Calculator result: 4" in result.stdout

    def test_configuration_flag_handling_streaming_enabled(self, cli_runner):
        """Streaming-enabled configuration produces rendered output."""
        result = _invoke_run(
            cli_runner,
            "Test streaming config",
            "Streaming configuration test completed",
        )

        assert result.exit_code == 0, result.stderr
        assert "Streaming configuration test completed" in result.stdout

    def test_configuration_flag_handling_backward_compatibility(self, cli_runner):
        """Backward-compatible configuration flags produce rendered output."""
        result = _invoke_run(
            cli_runner,
            "Test backward compatibility",
            "Backward compatibility test completed",
            extra_args=["--model", "gpt-4", "--format", "plain"],
        )

        assert result.exit_code == 0, result.stderr
        assert "Backward compatibility test completed" in result.stdout

    @pytest.mark.parametrize(
        ("agent_mode", "tool_mode"),
        [
            ("no_agent", "reasoning"),
            ("frame", "ptc"),
            ("frame", "classic"),
        ],
    )
    def test_configuration_flags_affect_streaming_behavior(
        self, cli_runner, agent_mode, tool_mode
    ):
        """Configuration flags are forwarded and output is rendered."""
        query = f"Test {agent_mode} {tool_mode}"
        final = f"Configuration test completed: {agent_mode} {tool_mode}"
        result = _invoke_run(
            cli_runner,
            query,
            final,
            extra_args=["--agent-mode", agent_mode, "--tool-call-mode", tool_mode],
        )

        assert result.exit_code == 0, result.stderr
        assert final in result.stdout
