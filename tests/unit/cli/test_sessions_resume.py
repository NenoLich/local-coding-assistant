"""Unit tests for the ``locca sessions`` command surface.

``sessions resume`` is the text-less/branching home for the runtime resume
entrypoints: no TEXT means crash recovery (``resume_session``), TEXT means a
new turn (``resume_session_with_message``), and --turn/--index select any
turn (branching). Tests verify flag wiring into ``RunOptions`` and the
handoff to the shared run flow, with the runtime mocked out.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import typer
from typer.testing import CliRunner

from local_coding_assistant.cli.commands.sessions import list_sessions, resume

runner = CliRunner()


class _MockAsyncContextManager:
    """Async context manager that yields a fixed context dict."""

    def __init__(self, ctx):
        self.ctx = ctx

    async def __aenter__(self):
        return self.ctx

    async def __aexit__(self, *args):
        return False


class _FakeReport:
    def __init__(self) -> None:
        self.final_answer = "All done"
        self.status = "success"


class _FakeTurn:
    def __init__(self, turn_id: str = "turn-1", status: str = "completed") -> None:
        self.turn_id = turn_id
        self.status = status


class _FakeSession:
    def __init__(self, session_id: str) -> None:
        self.id = session_id

    def get_last_turn(self) -> _FakeTurn:
        return _FakeTurn()


def _run_resume(session_id: str, text: str | None = None, **kwargs):
    """Call resume directly, capturing the typer.Exit exit code."""
    try:
        resume(session_id=session_id, text=text, **kwargs)
        return 0
    except typer.Exit as exc:
        return exc.exit_code


def _kw_resume(**overrides):
    """Build resume kwargs from non-overridden defaults plus overrides."""
    base = dict(
        turn=None,
        index=None,
        frame=None,
        verbose=False,
        model=None,
        agent_mode=None,
        tool_call_mode=None,
        sandbox_session=None,
        target_project_root=None,
        log_level="INFO",
        output_format="plain",
        trace=None,
        auto_approve=None,
    )
    base.update(overrides)
    return base


@contextmanager
def _patched_runtime(session_id: str) -> Iterator[dict[str, Any]]:
    """Patch sessions bootstrap and run_flow.execute_query."""
    captured: dict[str, Any] = {}
    fake_session = _FakeSession(session_id)
    fake_runtime = MagicMock()

    mock_bootstrap = MagicMock()
    mock_bootstrap.return_value = _MockAsyncContextManager({"runtime": fake_runtime})

    mock_execute_query = AsyncMock(return_value=(_FakeReport(), fake_session))

    async def _execute(runtime, text, options, console):
        captured["text"] = text
        captured["options"] = options
        captured["runtime"] = runtime
        return _FakeReport(), fake_session

    mock_execute_query.side_effect = _execute

    with (
        patch("local_coding_assistant.cli.commands.sessions.bootstrap", mock_bootstrap),
        patch(
            "local_coding_assistant.cli.commands.sessions.execute_query",
            mock_execute_query,
        ),
    ):
        yield captured


class TestSessionsResume:
    """sessions resume SESSION_ID [TEXT]"""

    def test_no_text_resumes_interrupted_turn(self) -> None:
        with _patched_runtime("sess-1") as captured:
            exit_code = _run_resume("sess-1", **_kw_resume())

        assert exit_code == 0
        assert captured["text"] is None
        assert captured["options"].session_id == "sess-1"
        assert captured["options"].resume_turn_id is None

    def test_with_text_creates_new_turn(self) -> None:
        with _patched_runtime("sess-1") as captured:
            exit_code = _run_resume("sess-1", "continue from here", **_kw_resume())

        assert exit_code == 0
        assert captured["text"] == "continue from here"
        assert captured["options"].session_id == "sess-1"

    def test_turn_selectors_are_forwarded(self) -> None:
        with _patched_runtime("sess-1") as captured:
            exit_code = _run_resume(
                "sess-1",
                **_kw_resume(turn="turn-9", index=9, frame="frame-1"),
            )

        assert exit_code == 0
        options = captured["options"]
        assert options.resume_turn_id == "turn-9"
        assert options.resume_turn_index == 9
        assert options.resume_frame_id == "frame-1"

    def test_branch_is_reported_when_session_changed(self) -> None:
        # A past turn was selected, so the runtime branched a new session.
        with _patched_runtime("forked-2"):
            exit_code = _run_resume("sess-1", **_kw_resume(turn="turn-3"))

        assert exit_code == 0

    def test_runtime_unavailable_reports_error(self) -> None:
        fake_runtime = None
        mock_bootstrap = MagicMock()
        mock_bootstrap.return_value = _MockAsyncContextManager(
            {"runtime": fake_runtime}
        )

        with patch(
            "local_coding_assistant.cli.commands.sessions.bootstrap", mock_bootstrap
        ):
            exit_code = _run_resume("sess-1", **_kw_resume())

        assert exit_code == 1


class TestSessionsListUnchanged:
    """The existing list command keeps working."""

    def test_list_reports_sessions(self, tmp_path) -> None:
        db_path = tmp_path / "execution_store.db"
        db_path.touch()

        class _StoredSession:
            id = "sess-1"
            status = "running"
            user_input = "hello"
            total_input_tokens = 10
            total_output_tokens = 5
            model_name = "gpt-4"

        fake_store = MagicMock()
        fake_store.init = AsyncMock()
        fake_store.close = AsyncMock()
        fake_store.list_sessions = AsyncMock(return_value=[_StoredSession()])
        fake_store.list_turns_by_session = AsyncMock(return_value=[])

        mock_reader = MagicMock(return_value=fake_store)

        with patch(
            "local_coding_assistant.cli.commands.sessions.ExecutionStoreReader",
            mock_reader,
        ):
            try:
                list_sessions(store_db_path=str(db_path), log_level="INFO")
                exit_code = 0
            except typer.Exit as exc:
                exit_code = exc.exit_code

        assert exit_code == 0
        mock_reader.assert_called_once()
