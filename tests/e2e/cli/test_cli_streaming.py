"""End-to-end tests for CLI run-query output (formerly streaming tests).

The refactored ``run query`` delegates execution to ``run_flow.execute_query``
and renders the resulting ``RunReport``. These tests mock the bootstrap /
execute-query boundary and verify that the rendered report content reaches
standard output.
"""

from unittest.mock import AsyncMock, MagicMock, patch

from local_coding_assistant.cli.main import app


class _MockAsyncContextManager:
    """Async context manager that yields a fixed context dict."""

    def __init__(self, ctx):
        self.ctx = ctx

    async def __aenter__(self):
        return self.ctx

    async def __aexit__(self, *args):
        return False


def _invoke_run(cli_runner, query, final_answer="[LLMService] Echo: response"):
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
        return cli_runner.invoke(app, ["run", "query", query])


class TestCLIStreamingE2E:
    """End-to-end tests for run-query output."""

    def test_run_query_output_contains_response(self, cli_runner):
        """The final answer is rendered to standard output."""
        result = _invoke_run(
            cli_runner, "Hello world", "[LLMService] Echo: Hello world!"
        )

        assert result.exit_code == 0, result.stderr
        assert "Hello world!" in result.stdout

    def test_run_query_tool_result(self, cli_runner):
        """A tool-assisted answer is rendered to standard output."""
        result = _invoke_run(cli_runner, "Calculate 5 + 3", "The result is 8")

        assert result.exit_code == 0, result.stderr
        assert "The result is 8" in result.stdout

    def test_run_query_multiple_iterations(self, cli_runner):
        """A multi-step answer is rendered to standard output."""
        result = _invoke_run(
            cli_runner,
            "Complex task requiring multiple steps",
            "Task completed in multiple steps",
        )

        assert result.exit_code == 0, result.stderr
        assert "Task completed in multiple steps" in result.stdout

    def test_run_query_error_message(self, cli_runner):
        """An error recovery message is rendered to standard output."""
        result = _invoke_run(
            cli_runner,
            "This will cause an error",
            "Sorry, I encountered an error. Please try again.",
        )

        assert result.exit_code == 0, result.stderr
        assert "Sorry, I encountered an error" in result.stdout

    def test_run_query_performance_content(self, cli_runner):
        """Longer streaming content is rendered to standard output."""
        result = _invoke_run(
            cli_runner, "Performance test", "Performance test completed"
        )

        assert result.exit_code == 0, result.stderr
        assert "Performance test completed" in result.stdout

    def test_run_query_progress_indicators(self, cli_runner):
        """Progress-tracking answer is rendered to standard output."""
        result = _invoke_run(
            cli_runner, "Test progress indicators", "Completed with progress tracking"
        )

        assert result.exit_code == 0, result.stderr
        assert "Completed with progress tracking" in result.stdout
