"""
Unit tests for the ``locca chat`` command.

Each console input becomes one turn dispatched through
``run_flow.execute_query`` with a mocked RuntimeManager. Console input is
scripted so each test exercises a single turn (or /exit), except where
multi-turn session continuity is explicitly tested.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from local_coding_assistant.cli.commands.chat import app, start

runner = CliRunner()

TEST_TEXT = "hello"
TEST_ANSWER = "42"


class _MockAsyncContextManager:
    """Async context manager that yields a fixed context dict."""

    def __init__(self, ctx):
        self.ctx = ctx

    async def __aenter__(self):
        return self.ctx

    async def __aexit__(self, *args):
        return False


_UNSET = object()


def _make_report(status="success", final_answer=TEST_ANSWER):
    report = MagicMock()
    report.final_answer = final_answer
    report.status = status
    return report


def _make_session(session_id="sess-1"):
    session = MagicMock()
    session.id = session_id
    return session


def _run_start(**kwargs):
    """Call start directly, capturing the typer.Exit exit code."""
    try:
        start(**kwargs)
        return 0
    except typer.Exit as exc:
        return exc.exit_code


def _kw(**overrides):
    """Build start kwargs from non-overridden defaults plus overrides."""
    base = dict(
        session_id=None,
        verbose=False,
        model=None,
        agent_mode="no_agent",
        tool_call_mode=None,
        target_project_root=None,
        log_level="INFO",
        auto_approve=None,
        trace=None,
        output_format="plain",
    )
    base.update(overrides)
    return base


@contextmanager
def _patched_chat(
    execute_query_return=None,
    runtime=_UNSET,
    input_values=None,
):
    """Patch bootstrap, execute_query, typer.echo and Console.input."""
    if execute_query_return is None:
        execute_query_return = (_make_report(), _make_session())
    actual_runtime = MagicMock() if runtime is _UNSET else runtime
    mock_ctx = {"runtime": actual_runtime}
    mock_bootstrap = MagicMock()
    mock_bootstrap.return_value = _MockAsyncContextManager(mock_ctx)
    mock_execute_query = AsyncMock(return_value=execute_query_return)
    mock_input = MagicMock(side_effect=input_values or [TEST_TEXT, "/exit"])
    mock_echo = MagicMock()
    with (
        patch("local_coding_assistant.cli.commands.chat.bootstrap", mock_bootstrap),
        patch(
            "local_coding_assistant.cli.commands.chat.execute_query",
            mock_execute_query,
        ),
        patch("local_coding_assistant.cli.commands.chat.typer.echo", mock_echo),
        patch("local_coding_assistant.cli.commands.chat.Console.input", mock_input),
    ):
        yield {
            "runtime": actual_runtime,
            "ctx": mock_ctx,
            "bootstrap": mock_bootstrap,
            "execute_query": mock_execute_query,
            "echo": mock_echo,
            "input": mock_input,
        }


class TestChatStart:
    """chat start SESSION_ID [options]"""

    def test_plain_success_prints_response(self):
        with _patched_chat() as m:
            exit_code = _run_start(**_kw())

        assert exit_code == 0
        m["echo"].assert_any_call("LOCCA chat. Type /exit to finish.")
        m["echo"].assert_any_call("\nResponse:")
        m["echo"].assert_any_call(TEST_ANSWER)
        m["execute_query"].assert_called_once()
        call_args = m["execute_query"].call_args
        assert call_args[0][0] is m["runtime"]
        assert call_args[0][1] == TEST_TEXT
        options = call_args[0][2]
        assert options.session_id is None
        assert options.model is None
        assert options.agent_mode == "no_agent"

    def test_plain_failure_returns_error_code(self):
        with _patched_chat(
            execute_query_return=(_make_report(status="error"), _make_session())
        ) as m:
            exit_code = _run_start(**_kw())

        assert exit_code == 1
        m["echo"].assert_any_call("\nResponse:")
        m["echo"].assert_any_call(TEST_ANSWER)
        err_calls = [c for c in m["echo"].call_args_list if c.kwargs.get("err")]
        assert not err_calls

    def test_exit_immediately_skips_execution(self):
        with _patched_chat(input_values=["/exit"]) as m:
            exit_code = _run_start(**_kw())

        assert exit_code == 0
        m["execute_query"].assert_not_called()

    def test_runtime_unavailable_reports_error(self):
        with _patched_chat(runtime=None) as m:
            exit_code = _run_start(**_kw())

        assert exit_code == 1
        m["echo"].assert_any_call(
            "Error: Runtime manager not available (LLM initialization failed)",
            err=True,
        )
        m["execute_query"].assert_not_called()
        m["echo"].assert_any_call(
            "Error: Runtime manager not available (LLM initialization failed)",
            err=True,
        )
        m["execute_query"].assert_not_called()

    def test_verbose_prints_session(self):
        with _patched_chat() as m:
            exit_code = _run_start(**_kw(verbose=True))

        assert exit_code == 0
        m["echo"].assert_any_call("[dim]session=sess-1[/dim]")

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
    def test_log_level_handling(self, log_level, expected_level):
        with _patched_chat() as m:
            _run_start(**_kw(log_level=log_level))

        m["bootstrap"].assert_called_once_with(
            log_level=expected_level, target_project_root=None
        )

    def test_model_agent_mode_tool_call_mode_forwarded(self):
        with _patched_chat() as m:
            _run_start(
                **_kw(
                    model="test-model",
                    agent_mode="frame_agent",
                    tool_call_mode="ptc",
                    verbose=True,
                )
            )

        options = m["execute_query"].call_args[0][2]
        assert options.model == "test-model"
        assert options.agent_mode == "frame_agent"
        assert options.tool_call_mode == "ptc"
        assert options.verbose is True

    def test_json_format_calls_render_report(self):
        with (
            _patched_chat(),
            patch(
                "local_coding_assistant.cli.commands.chat.render_report"
            ) as mock_render,
            patch(
                "local_coding_assistant.cli.commands.chat.coerce_report"
            ) as mock_coerce,
        ):
            mock_coerce.return_value = _make_report()
            exit_code = _run_start(**_kw(output_format="json"))

        assert exit_code == 0
        mock_render.assert_called_once()
        assert mock_render.call_args[0][0] is mock_coerce.return_value
        assert mock_render.call_args[1]["_format"] == "json"

    def test_trace_writes_trace_file(self, tmp_path):
        trace_file = tmp_path / "trace.json"
        with (
            _patched_chat() as m,
            patch("local_coding_assistant.cli.commands.chat.dump_report") as mock_dump,
        ):
            mock_dump.return_value = {"session_id": "sess-1", "turn": 1}
            _run_start(**_kw(output_format="plain", trace=str(trace_file)))

        assert trace_file.exists()
        mock_dump.assert_called_once()
        m["execute_query"].assert_called_once()

    def test_session_carried_across_turns(self):
        """Non-plain format keeps looping; session_id follows the last session."""
        session_1 = _make_session("sess-1")
        session_2 = _make_session("sess-2")
        mock_execute_query = AsyncMock(
            side_effect=[
                (_make_report(), session_1),
                (_make_report(), session_2),
            ]
        )
        mock_bootstrap = MagicMock()
        mock_bootstrap.return_value = _MockAsyncContextManager({"runtime": MagicMock()})
        mock_input = MagicMock(side_effect=[TEST_TEXT, TEST_TEXT, "/exit"])
        mock_echo = MagicMock()
        with (
            patch("local_coding_assistant.cli.commands.chat.bootstrap", mock_bootstrap),
            patch(
                "local_coding_assistant.cli.commands.chat.execute_query",
                mock_execute_query,
            ),
            patch("local_coding_assistant.cli.commands.chat.typer.echo", mock_echo),
            patch("local_coding_assistant.cli.commands.chat.Console.input", mock_input),
            patch("local_coding_assistant.cli.commands.chat.render_report"),
            patch(
                "local_coding_assistant.cli.commands.chat.coerce_report",
                return_value=_make_report(),
            ),
        ):
            exit_code = _run_start(**_kw(output_format="json"))

        assert exit_code == 0
        assert mock_execute_query.call_count == 2
        second_options = mock_execute_query.call_args_list[1][0][2]
        assert second_options.session_id == "sess-1"

    def test_safe_entrypoint_decorator(self):
        """safe_entrypoint is applied to the start function."""
        import inspect

        original_func = inspect.unwrap(start)
        assert original_func is not start
        assert original_func.__module__ == "local_coding_assistant.cli.commands.chat"

    def test_help_output(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "chat" in result.stdout.lower()
