"""Contract model validation & serialization tests."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from local_coding_assistant.storage.contracts import (
    MissingRefs,
    ResumeBatch,
    SessionReplay,
    StoredContextSnapshot,
    StoredEvent,
    StoredFrame,
    StoredSession,
    StoredTrace,
    StoredTurn,
    TurnReplay,
    _normalize_json_column,
)


class TestStoredSession:
    def test_required_fields_and_defaults(self) -> None:
        session = StoredSession(id="s1", started_at_ms=1000)
        assert session.id == "s1"
        assert session.started_at_ms == 1000
        assert session.status == "idle"
        assert session.user_input == ""
        assert session.agent_mode == "frame_agent"
        assert session.updated_at_ms is None
        assert session.model_name is None
        assert session.total_input_tokens == 0
        assert session.total_output_tokens == 0
        assert session.total_turns == 0
        assert session.total_frames == 0
        assert session.context_profile_json is None
        assert session.metadata_json is None

    def test_status_must_be_idle_or_running(self) -> None:
        with pytest.raises(ValidationError):
            StoredSession(id="s1", started_at_ms=1000, status="completed")
        with pytest.raises(ValidationError):
            StoredSession(id="s1", started_at_ms=1000, status="RUNNING")
        StoredSession(id="s1", started_at_ms=1000, status="idle")
        StoredSession(id="s1", started_at_ms=1000, status="running")

    def test_missing_required_fields_raises(self) -> None:
        with pytest.raises(ValidationError):
            StoredSession(started_at_ms=1000)
        with pytest.raises(ValidationError):
            StoredSession(id="s1")

    def test_json_serialization_roundtrip(self) -> None:
        session = StoredSession(
            id="s1",
            user_input="hi",
            agent_mode="frame_agent",
            status="running",
            started_at_ms=1000,
            updated_at_ms=2000,
            model_name="gpt-4",
            total_input_tokens=5,
            total_output_tokens=5,
            total_turns=1,
            total_frames=2,
            context_profile_json='{"a":1}',
            metadata_json='{"b":2}',
        )
        data = session.model_dump()
        restored = StoredSession.model_validate(data)
        assert restored == session
        assert restored.model_dump() == data
        assert restored.model_dump_json() == session.model_dump_json()
        again = StoredSession.model_validate_json(session.model_dump_json())
        assert again == session


class TestStoredTurn:
    def test_required_fields_and_defaults(self) -> None:
        turn = StoredTurn(
            id="t1",
            session_id="s1",
            trace_id="trace1",
            turn_index=0,
            user_input="query",
            config_snapshot_json="{}",
            started_at_ms=1000,
        )
        assert turn.id == "t1"
        assert turn.session_id == "s1"
        assert turn.trace_id == "trace1"
        assert turn.turn_index == 0
        assert turn.user_input == "query"
        assert turn.config_snapshot_json == "{}"
        assert turn.status == "running"
        assert turn.started_at_ms == 1000
        assert turn.finished_at_ms is None
        assert turn.paused_at_ms is None
        assert turn.pause_reason is None
        assert turn.resumed_at_ms is None
        assert turn.context_snapshot_id is None
        assert turn.total_input_tokens == 0
        assert turn.total_output_tokens == 0
        assert turn.total_frames == 0
        assert turn.message_boundary_json is None
        assert turn.metadata_json is None
        assert turn.final_answer is None

    def test_pause_resume_fields(self) -> None:
        turn = StoredTurn(
            id="t1",
            session_id="s1",
            trace_id="trace1",
            turn_index=0,
            user_input="query",
            config_snapshot_json="{}",
            started_at_ms=1000,
            status="paused",
            paused_at_ms=1500,
            pause_reason="approval",
            resumed_at_ms=2000,
            context_snapshot_id="snap1",
            message_boundary_json='{"session_id":"s1","start_seq":1,"end_seq":5}',
            metadata_json='{"k":"v"}',
        )
        assert turn.status == "paused"
        assert turn.paused_at_ms == 1500
        assert turn.pause_reason == "approval"
        assert turn.resumed_at_ms == 2000
        assert turn.context_snapshot_id == "snap1"
        boundary = json.loads(turn.message_boundary_json)
        assert boundary == {"session_id": "s1", "start_seq": 1, "end_seq": 5}
        assert json.loads(turn.metadata_json) == {"k": "v"}
        assert turn.final_answer is None

    def test_final_answer_field(self) -> None:
        """final_answer can be set on StoredTurn."""
        turn = StoredTurn(
            id="t1",
            session_id="s1",
            trace_id="trace1",
            turn_index=0,
            user_input="query",
            config_snapshot_json="{}",
            started_at_ms=1000,
            final_answer="The answer is 42",
        )
        assert turn.final_answer == "The answer is 42"

    def test_config_snapshot_json_required(self) -> None:
        with pytest.raises(ValidationError):
            StoredTurn(
                id="t1",
                session_id="s1",
                trace_id="trace1",
                turn_index=0,
                user_input="query",
                started_at_ms=1000,
            )


class TestStoredTrace:
    def test_required_fields(self) -> None:
        trace = StoredTrace(
            trace_id="trace1",
            session_id="s1",
            turn_id="t1",
        )
        assert trace.trace_id == "trace1"
        assert trace.session_id == "s1"
        assert trace.turn_id == "t1"

    def test_missing_required_raises(self) -> None:
        with pytest.raises(ValidationError):
            StoredTrace(trace_id="trace1", session_id="s1")


class TestStoredFrame:
    def test_required_fields_and_defaults(self) -> None:
        frame = StoredFrame(
            id="f1",
            session_id="s1",
            turn_id="t1",
            trace_id="trace1",
            iteration=1,
            started_at_ms=1000,
        )
        assert frame.status == "running_llm"
        assert frame.context_snapshot_id is None
        assert frame.finished_at_ms is None
        assert frame.duration_ms is None
        assert frame.input_tokens == 0
        assert frame.output_tokens == 0
        assert frame.model_name is None
        assert frame.finish_reason is None
        assert frame.error_message is None
        assert frame.pending_call_id is None
        assert frame.history_through_seq == 0
        assert frame.prompt_hash is None

    def test_optional_fields(self) -> None:
        frame = StoredFrame(
            id="f1",
            session_id="s1",
            turn_id="t1",
            trace_id="trace1",
            iteration=2,
            status="completed",
            context_snapshot_id="snap1",
            started_at_ms=1000,
            finished_at_ms=2000,
            duration_ms=1000,
            input_tokens=42,
            output_tokens=0,
            model_name="gpt-4",
            finish_reason="stop",
            error_message="boom",
            pending_call_id="call_1",
            history_through_seq=10,
            prompt_hash="sha256:abc",
        )
        assert frame.status == "completed"
        assert frame.input_tokens == 42
        assert frame.output_tokens == 0
        assert frame.history_through_seq == 10
        assert frame.prompt_hash == "sha256:abc"
        assert frame.pending_call_id == "call_1"


class TestStoredContextSnapshot:
    def test_required_fields_and_defaults(self) -> None:
        snap = StoredContextSnapshot(
            id="snap1",
            turn_id="t1",
            content_hash="hash",
            created_at_ms=1000,
        )
        assert snap.id == "snap1"
        assert snap.turn_id == "t1"
        assert snap.schema_version == 1
        assert snap.tool_set_ref is None
        assert snap.template_set_ref is None
        assert snap.repo_snapshot_ref is None
        assert snap.memory_snapshot_ref is None
        assert snap.message_boundary_json is None
        assert snap.content_hash == "hash"
        assert snap.payload_json is None
        assert snap.created_at_ms == 1000

    def test_to_runtime_with_all_refs(self) -> None:
        stored = StoredContextSnapshot(
            id="snap1",
            turn_id="t1",
            schema_version=1,
            tool_set_ref=json.dumps({"tool_ids": [1, 2], "tool_names": ["a", "b"]}),
            template_set_ref=json.dumps(
                {"template_ids": [3], "template_paths": ["x.jinja2"]}
            ),
            repo_snapshot_ref=json.dumps(
                {"target_project_root": "/root", "repo_snapshot_hash": "abc"}
            ),
            memory_snapshot_ref=json.dumps({"memory_ids": ["m1"]}),
            message_boundary_json=json.dumps(
                {"session_id": "s1", "start_seq": 1, "end_seq": 5}
            ),
            content_hash="hash",
            payload_json=None,
            created_at_ms=1000,
        )
        runtime = stored.to_runtime()
        assert runtime.id == "snap1"
        assert runtime.turn_id == "t1"
        assert runtime.schema_version == 1
        assert runtime.tool_set_ref.tool_ids == [1, 2]
        assert runtime.tool_set_ref.tool_names == ["a", "b"]
        assert runtime.tool_set_ref.version == "1.0.0"
        assert runtime.template_set_ref.template_ids == [3]
        assert runtime.template_set_ref.template_paths == ["x.jinja2"]
        assert runtime.repo_snapshot_ref is not None
        assert runtime.repo_snapshot_ref.target_project_root == "/root"
        assert runtime.repo_snapshot_ref.repo_snapshot_hash == "abc"
        assert runtime.memory_snapshot_ref is not None
        assert runtime.memory_snapshot_ref.memory_ids == ["m1"]
        assert runtime.message_boundary is not None
        assert runtime.message_boundary.session_id == "s1"
        assert runtime.message_boundary.start_seq == 1
        assert runtime.message_boundary.end_seq == 5
        assert runtime.content_hash == "hash"
        assert runtime.payload_json is None
        assert runtime.created_at_ms == 1000

    def test_to_runtime_normalizes_null_refs(self) -> None:
        stored = StoredContextSnapshot(
            id="snap1",
            turn_id="t1",
            tool_set_ref="null",
            template_set_ref='""',
            repo_snapshot_ref=None,
            memory_snapshot_ref="",
            message_boundary_json=None,
            content_hash="hash",
            created_at_ms=1000,
        )
        runtime = stored.to_runtime()
        assert runtime.tool_set_ref.tool_ids == []
        assert runtime.tool_set_ref.tool_names == []
        assert runtime.template_set_ref.template_ids == []
        assert runtime.template_set_ref.template_paths == []
        assert runtime.repo_snapshot_ref is None
        assert runtime.memory_snapshot_ref is None
        assert runtime.message_boundary is None

    def test_to_runtime_roundtrip_matches_from_stored_dict(self) -> None:
        from local_coding_assistant.runtime.context.profile import ContextSnapshot

        stored = StoredContextSnapshot(
            id="snap1",
            turn_id="t1",
            tool_set_ref=json.dumps({"tool_ids": [1], "tool_names": ["a"]}),
            repo_snapshot_ref=json.dumps(
                {"target_project_root": "/r", "repo_snapshot_hash": "h"}
            ),
            message_boundary_json=json.dumps(
                {"session_id": "s1", "start_seq": 1, "end_seq": 2}
            ),
            content_hash="hash",
            created_at_ms=1000,
        )
        runtime = stored.to_runtime()
        assert isinstance(runtime, ContextSnapshot)
        assert runtime.tool_set_ref.tool_ids == [1]
        assert runtime.repo_snapshot_ref.repo_snapshot_hash == "h"
        assert runtime.message_boundary.end_seq == 2


class TestStoredEvent:
    def test_required_fields(self) -> None:
        event = StoredEvent(
            trace_id="trace1",
            session_id="s1",
            turn_id="t1",
            frame_id="f1",
            seq=3,
            type="llm_complete",
            payload_json='{"content":"hi"}',
            timestamp_ms=1000,
        )
        assert event.trace_id == "trace1"
        assert event.session_id == "s1"
        assert event.turn_id == "t1"
        assert event.frame_id == "f1"
        assert event.seq == 3
        assert event.type == "llm_complete"
        assert event.payload_json == '{"content":"hi"}'
        assert event.timestamp_ms == 1000

    def test_turn_and_frame_optional(self) -> None:
        event = StoredEvent(
            trace_id="trace1",
            session_id="s1",
            seq=1,
            type="session_start",
            payload_json="{}",
            timestamp_ms=1000,
        )
        assert event.turn_id is None
        assert event.frame_id is None


class TestNormalizeJsonColumn:
    def test_none_returns_none(self) -> None:
        assert _normalize_json_column(None) is None

    def test_null_string_returns_none(self) -> None:
        assert _normalize_json_column("null") is None

    def test_empty_string_returns_none(self) -> None:
        assert _normalize_json_column("") is None

    def test_quoted_empty_string_returns_none(self) -> None:
        assert _normalize_json_column('""') is None

    def test_valid_json_passthrough(self) -> None:
        assert _normalize_json_column('{"a":1}') == '{"a":1}'
        assert _normalize_json_column("[]") == "[]"


class TestMissingRefs:
    def test_defaults(self) -> None:
        missing = MissingRefs()
        assert missing.tool_ids == []
        assert missing.template_ids == []
        assert missing.repo_hashes == []
        assert missing.memory_ids == []
        assert missing.is_empty is True

    def test_is_empty_false_when_any_field(self) -> None:
        assert MissingRefs(tool_ids=[1]).is_empty is False
        assert MissingRefs(template_ids=[1]).is_empty is False
        assert MissingRefs(repo_hashes=["x"]).is_empty is False
        assert MissingRefs(memory_ids=["x"]).is_empty is False

    def test_field_defaults_are_independent(self) -> None:
        a = MissingRefs()
        a.tool_ids.append(1)
        b = MissingRefs()
        assert b.tool_ids == []


class TestResumeBatchTurnReplaySessionReplay:
    def test_resume_batch_defaults(self) -> None:
        session = StoredSession(id="s1", started_at_ms=1000)
        turn = StoredTurn(
            id="t1",
            session_id="s1",
            trace_id="tr1",
            turn_index=0,
            user_input="q",
            config_snapshot_json="{}",
            started_at_ms=1000,
        )
        batch = ResumeBatch(session=session, turn=turn, snapshot=None)
        assert batch.session is session
        assert batch.turn is turn
        assert batch.snapshot is None
        assert batch.turns == []
        assert batch.frames == []
        assert batch.frame_events == {}
        assert batch.missing_refs.is_empty is True

    def test_resume_batch_with_all_fields(self) -> None:
        session = StoredSession(id="s1", started_at_ms=1000)
        turn = StoredTurn(
            id="t1",
            session_id="s1",
            trace_id="tr1",
            turn_index=0,
            user_input="q",
            config_snapshot_json="{}",
            started_at_ms=1000,
        )
        snap = StoredContextSnapshot(
            id="snap1",
            turn_id="t1",
            content_hash="hash",
            created_at_ms=1000,
        )
        frame = StoredFrame(
            id="f1",
            session_id="s1",
            turn_id="t1",
            trace_id="tr1",
            iteration=1,
            started_at_ms=1000,
        )
        batch = ResumeBatch(
            session=session,
            turn=turn,
            snapshot=snap,
            turns=[turn],
            frames=[frame],
            frame_events={"f1": [{"type": "x"}]},
            missing_refs=MissingRefs(repo_hashes=["missing"]),
        )
        assert batch.snapshot is snap
        assert batch.turns == [turn]
        assert batch.frames == [frame]
        assert batch.frame_events == {"f1": [{"type": "x"}]}
        assert batch.missing_refs.repo_hashes == ["missing"]
        assert batch.missing_refs.is_empty is False

    def test_turn_replay_defaults(self) -> None:
        turn = StoredTurn(
            id="t1",
            session_id="s1",
            trace_id="tr1",
            turn_index=0,
            user_input="q",
            config_snapshot_json="{}",
            started_at_ms=1000,
        )
        replay = TurnReplay(turn=turn, snapshot=None)
        assert replay.turn is turn
        assert replay.snapshot is None
        assert replay.frames == []
        assert replay.frame_events == {}
        assert replay.messages == []
        assert replay.missing_refs.is_empty is True

    def test_session_replay_defaults(self) -> None:
        session = StoredSession(id="s1", started_at_ms=1000)
        replay = SessionReplay(session=session)
        assert replay.session is session
        assert replay.turns == []

    def test_dataclasses_use_slots(self) -> None:
        assert hasattr(ResumeBatch, "__slots__")
        assert hasattr(TurnReplay, "__slots__")
        assert hasattr(SessionReplay, "__slots__")
