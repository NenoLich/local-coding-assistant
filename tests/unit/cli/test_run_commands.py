"""
Unit tests for CLI run commands.

The refactored ``run query`` command delegates execution to
``run_flow.execute_query`` and renders the resulting ``RunReport`` via the
rendering registry, instead of calling ``runtime.orchestrate`` directly.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from local_coding_assistant.cli.commands.run import query

runner = CliRunner()

TEST_QUERY = "What is the meaning of life?"
TEST_MODEL = "test-model"
TEST_ANSWER = "42"


class _MockAsyncContextManager:
    """Async context manager that yields a fixed context dict."""

    def __init__(self, ctx):
        self.ctx = ctx

    async def __aenter__(self):
        return self.ctx

    async def __aexit__(self, *args):
        return False


def _make_report(final_answer=TEST_ANSWER, status="success"):
    report = MagicMock()
    report.final_answer = final_answer
    report.status = status
    return report


def _make_session(session_id="sess-1"):
    session = MagicMock()
    session.id = session_id
    return session


def _run_query(**kwargs):
    """Call query directly, capturing the typer.Exit exit code."""
    try:
        query(**kwargs)
        return 0
    except typer.Exit as exc:
        return exc.exit_code


@pytest.fixture
def setup_mocks():
    """Patch bootstrap, execute_query and typer.echo for run query tests."""
    mock_runtime = MagicMock()
    mock_runtime.event_pipeline = MagicMock()
    mock_ctx = {"runtime": mock_runtime}

    mock_bootstrap = MagicMock()
    mock_bootstrap.return_value = _MockAsyncContextManager(mock_ctx)

    mock_execute_query = AsyncMock(return_value=(_make_report(), _make_session()))

    mock_echo = MagicMock()

    with (
        patch("local_coding_assistant.cli.commands.run.bootstrap", mock_bootstrap),
        patch(
            "local_coding_assistant.cli.commands.run.execute_query",
            mock_execute_query,
        ),
        patch("local_coding_assistant.cli.commands.run.typer.echo", mock_echo),
    ):
        yield {
            "runtime": mock_runtime,
            "ctx": mock_ctx,
            "bootstrap": mock_bootstrap,
            "execute_query": mock_execute_query,
            "echo": mock_echo,
        }


def _kw(**overrides):
    """Build query kwargs from non-overridden defaults plus overrides."""
    base = dict(
        verbose=False,
        model=None,
        agent_mode="no_agent",
        tool_call_mode=None,
        sandbox_session=None,
        target_project_root=None,
        log_level="INFO",
        output_format="plain",
        session_id=None,
        trace=None,
        auto_approve=None,
    )
    base.update(overrides)
    return base


class TestRunCommands:
    """Test run command functionality."""

    def test_query_basic(self, setup_mocks):
        """Basic query prints the response and returns success."""
        mocks = setup_mocks
        mocks["execute_query"].return_value = (_make_report(), _make_session())

        exit_code = _run_query(text=TEST_QUERY, **_kw())

        assert exit_code == 0
        mocks["echo"].assert_any_call("\nResponse:")
        mocks["echo"].assert_any_call(TEST_ANSWER)
        mocks["bootstrap"].assert_called_once_with(
            log_level=logging.INFO, target_project_root=None
        )
        mocks["execute_query"].assert_called_once()
        call_args = mocks["execute_query"].call_args
        assert call_args[0][0] is mocks["runtime"]
        assert call_args[0][1] == TEST_QUERY
        options = call_args[0][2]
        assert options.model is None
        assert options.agent_mode == "no_agent"
        assert options.tool_call_mode is None
        assert options.session_id is None
        assert options.verbose is False

    def test_query_with_model(self, setup_mocks):
        """Query forwards the model override to RunOptions."""
        _run_query(text=TEST_QUERY, **_kw(model=TEST_MODEL))

        options = setup_mocks["execute_query"].call_args[0][2]
        assert options.model == TEST_MODEL

    def test_verbose_mode_prints_session(self, setup_mocks):
        """Verbose mode prints the session id."""
        mocks = setup_mocks
        mocks["execute_query"].return_value = (
            _make_report(),
            _make_session("sess-42"),
        )

        exit_code = _run_query(text=TEST_QUERY, **_kw(verbose=True))

        assert exit_code == 0
        mocks["echo"].assert_any_call("session=sess-42")

    def test_log_level_parameter(self, setup_mocks):
        """Log level is forwarded to bootstrap."""
        _run_query(text=TEST_QUERY, **_kw(log_level="DEBUG"))

        setup_mocks["bootstrap"].assert_called_once_with(
            log_level=logging.DEBUG, target_project_root=None
        )

    def test_runtime_unavailable_reports_error(self, setup_mocks):
        """Missing runtime produces an error exit code."""
        mocks = setup_mocks
        mocks["ctx"]["runtime"] = None

        with patch("local_coding_assistant.cli.commands.run.typer.echo") as mock_echo:
            exit_code = _run_query(text=TEST_QUERY, **_kw())

        assert exit_code == 1
        mock_echo.assert_any_call(
            "Error: Runtime manager not available (LLM initialization failed)",
            err=True,
        )
        mocks["execute_query"].assert_not_called()

    def test_no_terminal_report_reports_error(self, setup_mocks):
        """None report produces an error exit code."""
        mocks = setup_mocks
        mocks["execute_query"].return_value = (None, _make_session())

        exit_code = _run_query(text=TEST_QUERY, **_kw())

        assert exit_code == 1
        mocks["echo"].assert_any_call(
            "Error: No terminal report received from runtime", err=True
        )

    def test_safe_entrypoint_decorator(self):
        """safe_entrypoint is applied to the query function."""
        import inspect

        original_func = inspect.unwrap(query)
        assert original_func is not query
        assert original_func.__module__ == "local_coding_assistant.cli.commands.run"

    @pytest.mark.parametrize(
        ("log_level", "expected_level"),
        [
            ("DEBUG", logging.DEBUG),
            ("INFO", logging.INFO),
            ("WARNING", logging.WARNING),
            ("ERROR", logging.ERROR),
            ("invalid", logging.INFO),
        ],
    )
    def test_log_level_handling(self, log_level, expected_level, setup_mocks):
        """Different log level strings resolve to logging constants."""
        _run_query(text=TEST_QUERY, **_kw(log_level=log_level))

        setup_mocks["bootstrap"].assert_called_once_with(
            log_level=expected_level, target_project_root=None
        )

    def test_json_format_calls_render_report(self, setup_mocks):
        """Non-plain formats delegate to render_report."""
        mocks = setup_mocks
        mock_report = _make_report()
        with (
            patch(
                "local_coding_assistant.cli.commands.run.render_report"
            ) as mock_render,
            patch(
                "local_coding_assistant.cli.commands.run.coerce_report"
            ) as mock_coerce,
        ):
            mock_coerce.return_value = mock_report
            mocks["execute_query"].return_value = (mock_report, _make_session())

            kws = _kw()
            kws["output_format"] = "json"
            exit_code = _run_query(text=TEST_QUERY, **kws)

            assert exit_code == 0
            mock_render.assert_called_once()
            assert mock_render.call_args[0][0] is mock_report
            assert mock_render.call_args[1]["_format"] == "json"
