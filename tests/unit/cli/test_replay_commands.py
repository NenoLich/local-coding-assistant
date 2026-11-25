"""
Unit tests for the ``locca replay`` command.

The replay command reads a stored session graph from the execution store and
renders it as plain text or JSON. Tests mock ``ExecutionStoreReader`` and
exercise the show command plus the module-level rendering helpers.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import typer
from typer.testing import CliRunner

from local_coding_assistant.cli.commands.replay import (
    _DEFAULT_CONTENT_MAX_LEN,
    _filter_turn,
    _missing_refs_summary,
    _parse_json_field,
    _render_event,
    _render_json,
    _truncate,
    app,
    show,
)
from local_coding_assistant.storage.contracts import (
    MissingRefs,
    SessionReplay,
    StoredFrame,
    StoredSession,
    StoredTurn,
    TurnReplay,
)

runner = CliRunner()


def _make_session_replay():
    session = StoredSession(
        id="sess-1",
        user_input="hello world",
        status="idle",
        started_at_ms=1000,
        model_name="mock-model",
        total_input_tokens=10,
        total_output_tokens=5,
        total_turns=1,
        total_frames=1,
    )
    turn = StoredTurn(
        id="turn-1",
        session_id="sess-1",
        trace_id="trace-1",
        turn_index=0,
        user_input="hello world",
        config_snapshot_json=None,
        started_at_ms=1001,
        status="completed",
        total_input_tokens=10,
        total_output_tokens=5,
        total_frames=1,
    )
    frame = StoredFrame(
        id="frame-1",
        session_id="sess-1",
        turn_id="turn-1",
        trace_id="trace-1",
        iteration=1,
        status="completed",
        started_at_ms=1002,
        input_tokens=5,
        output_tokens=5,
    )
    turn_replay = TurnReplay(
        turn=turn,
        snapshot=None,
        frames=[frame],
        frame_events={
            "frame-1": [
                {
                    "id": "e1",
                    "trace_id": "trace-1",
                    "session_id": "sess-1",
                    "turn_id": "turn-1",
                    "frame_id": "frame-1",
                    "seq": 1,
                    "type": "LLM_CHUNK",
                    "data_json": '{"content": "hi"}',
                    "timestamp_ms": 1,
                    "timestamp": "2026-01-01T00:00:00",
                }
            ]
        },
        messages=[
            {
                "id": "m1",
                "seq": 1,
                "role": "assistant",
                "content": "hi there",
                "turn_id": "turn-1",
                "tool_call_id": None,
                "tool_name": None,
                "metadata_json": None,
                "dedupe_key": None,
                "created_at": "2026-01-01T00:00:00",
            }
        ],
        missing_refs=MissingRefs(),
    )
    return SessionReplay(session=session, turns=[turn_replay])


def _run_show(**kwargs):
    """Call show directly, capturing the typer.Exit exit code.

    Typer replaces option defaults with OptionInfo objects when the command
    is invoked directly, so all options are filled in explicitly.
    """
    defaults = dict(
        turn_id=None,
        frame_id=None,
        events=False,
        output_format="plain",
        content_max_len=_DEFAULT_CONTENT_MAX_LEN,
        log_level="INFO",
    )
    defaults.update(kwargs)
    try:
        show(**defaults)
        return 0
    except typer.Exit as exc:
        return exc.exit_code


@contextmanager
def _patched_store(replay=None):
    """Patch ExecutionStoreReader, setup_logging and typer.echo."""
    mock_store = MagicMock()
    mock_store.init = AsyncMock()
    mock_store.close = AsyncMock()
    mock_store.load_replay_batch = AsyncMock(return_value=replay)
    mock_reader = MagicMock(return_value=mock_store)
    mock_echo = MagicMock()
    with (
        patch(
            "local_coding_assistant.cli.commands.replay.ExecutionStoreReader",
            mock_reader,
        ),
        patch("local_coding_assistant.cli.commands.replay.setup_logging"),
        patch("local_coding_assistant.cli.commands.replay.typer.echo", mock_echo),
    ):
        yield {"store": mock_store, "reader": mock_reader, "echo": mock_echo}


class TestReplayShow:
    """replay show SESSION_ID [options]"""

    def test_db_not_found_returns_error(self, tmp_path):
        missing_db = tmp_path / "does_not_exist.db"
        exit_code = _run_show(session_id="sess-1", store_db_path=str(missing_db))

        assert exit_code == 1

    def test_session_not_found_returns_error(self, tmp_path):
        db_path = tmp_path / "store.db"
        db_path.touch()
        with _patched_store(replay=None) as m:
            exit_code = _run_show(session_id="sess-1", store_db_path=str(db_path))

        assert exit_code == 1
        m["echo"].assert_any_call("Session not found: sess-1", err=True)
        m["store"].load_replay_batch.assert_called_once_with("sess-1")

    def test_turn_not_found_returns_error(self, tmp_path):
        db_path = tmp_path / "store.db"
        db_path.touch()
        with _patched_store(replay=_make_session_replay()) as m:
            exit_code = _run_show(
                session_id="sess-1", turn_id="nope", store_db_path=str(db_path)
            )

        assert exit_code == 1
        m["echo"].assert_any_call("Turn not found: nope", err=True)

    def test_plain_render(self, tmp_path):
        db_path = tmp_path / "store.db"
        db_path.touch()
        replay = _make_session_replay()
        with _patched_store(replay=replay) as m:
            exit_code = _run_show(session_id="sess-1", store_db_path=str(db_path))

        assert exit_code == 0
        echoed = " ".join(str(call.args[0]) for call in m["echo"].call_args_list)
        assert "session sess-1  status=idle" in echoed
        assert "input: hello world" in echoed
        assert "turn turn-1" in echoed
        assert "frame frame-1" in echoed
        assert "assistant: hi there" in echoed
        m["store"].init.assert_awaited_once()
        m["store"].close.assert_awaited_once()

    def test_json_render(self, tmp_path):
        db_path = tmp_path / "store.db"
        db_path.touch()
        replay = _make_session_replay()
        with _patched_store(replay=replay) as m:
            exit_code = _run_show(
                session_id="sess-1",
                output_format="json",
                store_db_path=str(db_path),
            )

        assert exit_code == 0
        json_calls = [
            str(call.args[0])
            for call in m["echo"].call_args_list
            if call.args and call.args[0].strip().startswith("{")
        ]
        assert json_calls, "expected a JSON echo call"
        import json as json_module

        payload = json_module.loads(json_calls[0])
        assert payload["session"]["id"] == "sess-1"
        assert payload["session"]["user_input"] == "hello world"
        assert len(payload["turns"]) == 1
        assert payload["turns"][0]["frames"][0]["id"] == "frame-1"
        assert payload["turns"][0]["messages"][0]["role"] == "assistant"

    def test_turn_id_filters_turns(self, tmp_path):
        db_path = tmp_path / "store.db"
        db_path.touch()
        replay = _make_session_replay()
        with (
            _patched_store(replay=replay),
            patch(
                "local_coding_assistant.cli.commands.replay._filter_turn",
                return_value=replay,
            ) as mock_filter,
        ):
            _run_show(
                session_id="sess-1",
                turn_id="turn-1",
                store_db_path=str(db_path),
            )

        mock_filter.assert_called_once_with(replay, "turn-1")

    def test_content_max_len_truncates(self, tmp_path):
        db_path = tmp_path / "store.db"
        db_path.touch()
        replay = _make_session_replay()
        with _patched_store(replay=replay) as m:
            _run_show(
                session_id="sess-1",
                content_max_len=5,
                store_db_path=str(db_path),
            )

        echoed = " ".join(str(call.args[0]) for call in m["echo"].call_args_list)
        assert "input: hello" in echoed

    def test_with_events_renders_events(self, tmp_path):
        db_path = tmp_path / "store.db"
        db_path.touch()
        replay = _make_session_replay()
        with _patched_store(replay=replay) as m:
            _run_show(
                session_id="sess-1",
                events=True,
                store_db_path=str(db_path),
            )

        echoed = " ".join(str(call.args[0]) for call in m["echo"].call_args_list)
        assert "LLM_CHUNK" in echoed
        assert '"content": "hi"' in echoed

    def test_frame_id_filters_frames(self, tmp_path):
        db_path = tmp_path / "store.db"
        db_path.touch()
        replay = _make_session_replay()
        with _patched_store(replay=replay) as m:
            _run_show(
                session_id="sess-1",
                frame_id="frame-1",
                store_db_path=str(db_path),
            )

        echoed = " ".join(str(call.args[0]) for call in m["echo"].call_args_list)
        assert "frame frame-1" in echoed

    def test_log_level_forwards_to_setup_logging(self):
        import logging as logging_module

        replay = _make_session_replay()
        mock_store = MagicMock()
        mock_store.init = AsyncMock()
        mock_store.close = AsyncMock()
        mock_store.load_replay_batch = AsyncMock(return_value=replay)
        mock_reader = MagicMock(return_value=mock_store)
        mock_setup = MagicMock()
        mock_echo = MagicMock()
        with (
            patch(
                "local_coding_assistant.cli.commands.replay.ExecutionStoreReader",
                mock_reader,
            ),
            patch(
                "local_coding_assistant.cli.commands.replay.setup_logging",
                mock_setup,
            ),
            patch("local_coding_assistant.cli.commands.replay.typer.echo", mock_echo),
        ):
            _run_show(session_id="sess-1", log_level="DEBUG")

        mock_setup.assert_called_once()
        assert mock_setup.call_args[1]["level"] == logging_module.DEBUG

    def test_invalid_log_level_defaults_to_info(self):
        import logging as logging_module

        replay = _make_session_replay()
        mock_store = MagicMock()
        mock_store.init = AsyncMock()
        mock_store.close = AsyncMock()
        mock_store.load_replay_batch = AsyncMock(return_value=replay)
        mock_reader = MagicMock(return_value=mock_store)
        mock_setup = MagicMock()
        mock_echo = MagicMock()
        with (
            patch(
                "local_coding_assistant.cli.commands.replay.ExecutionStoreReader",
                mock_reader,
            ),
            patch(
                "local_coding_assistant.cli.commands.replay.setup_logging",
                mock_setup,
            ),
            patch("local_coding_assistant.cli.commands.replay.typer.echo", mock_echo),
        ):
            _run_show(session_id="sess-1", log_level="bogus")

        mock_setup.assert_called_once()
        assert mock_setup.call_args[1]["level"] == logging_module.INFO

    def test_safe_entrypoint_decorator(self):
        import inspect

        original_func = inspect.unwrap(show)
        assert original_func is not show
        assert original_func.__module__ == "local_coding_assistant.cli.commands.replay"

    def test_help_output(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "replay" in result.stdout.lower()


class TestReplayHelpers:
    """Module-level rendering helpers."""

    def test_filter_turn_keeps_matching_turn(self):
        turn_a = StoredTurn(
            id="turn-a",
            session_id="sess-1",
            trace_id="trace-1",
            turn_index=0,
            user_input="a",
            config_snapshot_json=None,
            started_at_ms=1,
        )
        turn_b = StoredTurn(
            id="turn-b",
            session_id="sess-1",
            trace_id="trace-1",
            turn_index=1,
            user_input="b",
            config_snapshot_json=None,
            started_at_ms=2,
        )
        replay = SessionReplay(
            session=StoredSession(
                id="sess-1",
                user_input="x",
                status="idle",
                started_at_ms=1,
            ),
            turns=[
                TurnReplay(turn=turn_a, snapshot=None),
                TurnReplay(turn=turn_b, snapshot=None),
            ],
        )

        filtered = _filter_turn(replay, "turn-b")
        assert filtered is not None
        assert len(filtered.turns) == 1
        assert filtered.turns[0].turn.id == "turn-b"

        assert _filter_turn(replay, "nope") is None

    def test_truncate(self):
        assert _truncate(None, 10) == ""
        assert _truncate("hello", 0) == "hello"
        assert _truncate("hello", -1) == "hello"
        assert _truncate("hello world", 5) == "hello"
        assert _truncate("hi", 10) == "hi"

    def test_missing_refs_summary(self):
        assert _missing_refs_summary(MissingRefs()) == "none"
        summary = _missing_refs_summary(
            MissingRefs(tool_ids=[1, 2], repo_hashes=["abc"])
        )
        assert "tools=[1, 2]" in summary
        assert "repo=['abc']" in summary

    def test_parse_json_field(self):
        assert _parse_json_field(None) is None
        assert _parse_json_field("") == ""
        assert _parse_json_field('{"a": 1}') == {"a": 1}
        assert _parse_json_field("not json") == "not json"

    def test_render_event(self):
        event = {
            "id": "e1",
            "trace_id": "trace-1",
            "session_id": "sess-1",
            "turn_id": "turn-1",
            "frame_id": "frame-1",
            "seq": 1,
            "type": "LLM_CHUNK",
            "data_json": '{"content": "hi"}',
            "timestamp_ms": 1,
            "timestamp": "2026-01-01T00:00:00",
        }
        rendered = _render_event(event, content_max_len=0)
        assert rendered["id"] == "e1"
        assert rendered["type"] == "LLM_CHUNK"
        assert rendered["data_json"] == {"content": "hi"}

    def test_render_json_structure(self):
        replay = _make_session_replay()
        rendered = _render_json(replay, with_events=False, content_max_len=0)
        assert rendered["session"]["id"] == "sess-1"
        assert rendered["session"]["status"] == "idle"
        assert rendered["session"]["user_input"] == "hello world"
        assert rendered["session"]["total_turns"] == 1
        turn = rendered["turns"][0]
        assert turn["id"] == "turn-1"
        assert turn["turn_index"] == 0
        assert turn["snapshot"] is None
        assert turn["missing_refs"] == {
            "tool_ids": [],
            "template_ids": [],
            "repo_hashes": [],
        }
        assert turn["frames"][0]["id"] == "frame-1"
        assert turn["frames"][0]["events"] == []
        assert turn["messages"][0]["role"] == "assistant"
        assert turn["messages"][0]["content"] == "hi there"
