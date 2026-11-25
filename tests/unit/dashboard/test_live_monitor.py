"""
Unit tests for the live monitoring plane: LiveMonitor and EventProjector.

These seal the core behaviour of reconstructing in-memory session/turn/frame
state from runtime events ingested via the dashboard ingest endpoint.
"""

from datetime import UTC, datetime

from local_coding_assistant.dashboard.services.live_monitor import LiveMonitor
from local_coding_assistant.runtime.events.contracts import (
    ErrorEvent,
    EventType,
    FrameCompleteEvent,
    FrameStartEvent,
    LLMChunkEvent,
    LLMCompleteEvent,
    SessionBranchEvent,
    SessionInterruptEvent,
    SessionResumeEvent,
    SessionStartEvent,
    ToolResultEvent,
    ToolStartEvent,
    TurnCompleteEvent,
    TurnStartEvent,
)


def _ts(seconds: int = 0) -> datetime:
    return datetime(2024, 1, 1, 12, 0, seconds, tzinfo=UTC)


def _seed_session_and_turn(live_monitor, make_runtime_event):
    live_monitor.ingest_event(
        make_runtime_event(
            SessionStartEvent,
            session_id="session-1",
            user_input="hello",
            agent_mode="frame_agent",
            timestamp=_ts(1),
        )
    )
    live_monitor.ingest_event(
        make_runtime_event(
            TurnStartEvent,
            session_id="session-1",
            turn_id="turn-1",
            turn_index=0,
            user_input="hello",
            timestamp=_ts(2),
        )
    )


def _seed_session_turn_frame(monitor, make_runtime_event):
    monitor.ingest_event(
        make_runtime_event(
            SessionStartEvent,
            session_id="session-1",
            user_input="a",
            agent_mode="frame_agent",
            timestamp=_ts(1),
        )
    )
    monitor.ingest_event(
        make_runtime_event(
            TurnStartEvent,
            session_id="session-1",
            turn_id="turn-1",
            turn_index=0,
            user_input="a",
            timestamp=_ts(2),
        )
    )
    monitor.ingest_event(
        make_runtime_event(
            FrameStartEvent,
            session_id="session-1",
            turn_id="turn-1",
            frame_id="frame-1",
            iteration=0,
            context_snapshot_id="snap-1",
            context_hash="hash-1",
            timestamp=_ts(3),
        )
    )


class TestSessionLifecycle:
    def test_session_start_creates_running_session(
        self, live_monitor, make_runtime_event
    ):
        event = make_runtime_event(
            SessionStartEvent,
            session_id="session-1",
            user_input="hello",
            agent_mode="frame_agent",
            model_name="gpt-4",
            timestamp=_ts(1),
        )
        live_monitor.ingest_event(event)

        session = live_monitor.get_session("session-1")
        assert session is not None
        assert session.session_id == "session-1"
        assert session.status == "running"
        assert session.user_input == "hello"
        assert session.agent_mode == "frame_agent"
        assert session.model_name == "gpt-4"

    def test_session_start_updates_existing_session(
        self, live_monitor, make_runtime_event
    ):
        live_monitor.ingest_event(
            make_runtime_event(
                SessionStartEvent,
                session_id="session-1",
                user_input="hello",
                agent_mode="frame_agent",
                timestamp=_ts(1),
            )
        )
        live_monitor.ingest_event(
            make_runtime_event(
                SessionStartEvent,
                session_id="session-1",
                user_input="hello again",
                agent_mode="chat",
                timestamp=_ts(2),
            )
        )

        session = live_monitor.get_session("session-1")
        assert session.user_input == "hello again"
        assert session.agent_mode == "chat"

    def test_session_interrupt_marks_interrupted(
        self, live_monitor, make_runtime_event
    ):
        live_monitor.ingest_event(
            make_runtime_event(
                SessionStartEvent,
                session_id="session-1",
                user_input="hello",
                agent_mode="frame_agent",
                timestamp=_ts(1),
            )
        )
        live_monitor.ingest_event(
            make_runtime_event(
                SessionInterruptEvent,
                session_id="session-1",
                turn_id="turn-1",
                reason="interrupt",
                timestamp=_ts(2),
            )
        )

        session = live_monitor.get_session("session-1")
        assert session.status == "interrupted"

    def test_session_resume_marks_running(self, live_monitor, make_runtime_event):
        live_monitor.ingest_event(
            make_runtime_event(
                SessionStartEvent,
                session_id="session-1",
                user_input="hello",
                agent_mode="frame_agent",
                timestamp=_ts(1),
            )
        )
        live_monitor.ingest_event(
            make_runtime_event(
                SessionInterruptEvent,
                session_id="session-1",
                turn_id="turn-1",
                timestamp=_ts(2),
            )
        )
        live_monitor.ingest_event(
            make_runtime_event(
                SessionResumeEvent,
                session_id="session-1",
                timestamp=_ts(3),
            )
        )

        session = live_monitor.get_session("session-1")
        assert session.status == "running"

    def test_session_branch_creates_new_session(self, live_monitor, make_runtime_event):
        live_monitor.ingest_event(
            make_runtime_event(
                SessionStartEvent,
                session_id="session-1",
                user_input="hello",
                agent_mode="frame_agent",
                timestamp=_ts(1),
            )
        )
        live_monitor.ingest_event(
            make_runtime_event(
                SessionBranchEvent,
                session_id="session-1",
                parent_session_id="session-1",
                branch_from_turn_id="turn-1",
                new_session_id="session-2",
                turn_id="turn-1",
                timestamp=_ts(2),
            )
        )

        assert live_monitor.get_session("session-2") is not None
        assert live_monitor.get_session("session-2").status == "running"


class TestTurnLifecycle:
    def test_turn_start_creates_turn_in_session(self, live_monitor, make_runtime_event):
        _seed_session_and_turn(live_monitor, make_runtime_event)

        session = live_monitor.get_session("session-1")
        turn = session.turns.get("turn-1")
        assert turn is not None
        assert turn.turn_id == "turn-1"
        assert turn.turn_index == 0
        assert turn.status == "running"
        assert turn.user_input == "hello"

    def test_turn_start_creates_session_and_turn_without_prior_session_start(
        self, live_monitor, make_runtime_event
    ):
        # The projector tolerantly creates the session on demand.
        live_monitor.ingest_event(
            make_runtime_event(
                TurnStartEvent,
                session_id="session-1",
                turn_id="turn-1",
                turn_index=0,
                user_input="hello",
                timestamp=_ts(2),
            )
        )
        session = live_monitor.get_session("session-1")
        assert session is not None
        assert "turn-1" in session.turns
        assert session.turns["turn-1"].status == "running"

    def test_turn_complete_updates_turn_status(self, live_monitor, make_runtime_event):
        _seed_session_and_turn(live_monitor, make_runtime_event)
        live_monitor.ingest_event(
            make_runtime_event(
                TurnCompleteEvent,
                session_id="session-1",
                turn_id="turn-1",
                status="completed",
                final_answer="done",
                total_frames=2,
                timestamp=_ts(3),
            )
        )

        turn = live_monitor.get_session("session-1").turns["turn-1"]
        assert turn.status == "completed"
        assert turn.final_answer == "done"
        assert turn.total_frames == 2
        assert turn.finished_at is not None


class TestFrameLifecycle:
    def test_frame_start_creates_frame_in_turn(self, live_monitor, make_runtime_event):
        _seed_session_and_turn(live_monitor, make_runtime_event)
        live_monitor.ingest_event(
            make_runtime_event(
                FrameStartEvent,
                session_id="session-1",
                turn_id="turn-1",
                frame_id="frame-1",
                iteration=0,
                context_snapshot_id="snap-1",
                context_hash="hash-1",
                timestamp=_ts(3),
            )
        )

        turn = live_monitor.get_session("session-1").turns["turn-1"]
        frame = turn.frames.get("frame-1")
        assert frame is not None
        assert frame.frame_id == "frame-1"
        assert frame.iteration == 0
        assert frame.status == "running_llm"
        assert turn.total_frames == 1

    def test_frame_complete_updates_frame(self, live_monitor, make_runtime_event):
        _seed_session_and_turn(live_monitor, make_runtime_event)
        live_monitor.ingest_event(
            make_runtime_event(
                FrameStartEvent,
                session_id="session-1",
                turn_id="turn-1",
                frame_id="frame-1",
                iteration=0,
                context_snapshot_id="snap-1",
                context_hash="hash-1",
                timestamp=_ts(3),
            )
        )
        live_monitor.ingest_event(
            make_runtime_event(
                FrameCompleteEvent,
                session_id="session-1",
                turn_id="turn-1",
                frame_id="frame-1",
                status="completed",
                finish_reason="stop",
                duration_ms=2000,
                model_name="gpt-4",
                tool_call_count=3,
                timestamp=_ts(5),
            )
        )

        frame = live_monitor.get_session("session-1").turns["turn-1"].frames["frame-1"]
        assert frame.status == "completed"
        assert frame.finish_reason == "stop"
        assert frame.duration_ms == 2000
        assert frame.model_name == "gpt-4"
        assert frame.tool_call_count == 3

    def test_frame_complete_unknown_frame_is_ignored(
        self, live_monitor, make_runtime_event
    ):
        _seed_session_and_turn(live_monitor, make_runtime_event)
        live_monitor.ingest_event(
            make_runtime_event(
                FrameCompleteEvent,
                session_id="session-1",
                turn_id="turn-1",
                frame_id="nonexistent",
                status="completed",
                timestamp=_ts(5),
            )
        )
        turn = live_monitor.get_session("session-1").turns["turn-1"]
        assert turn.frames == {}


class TestLLMEvents:
    def test_llm_complete_updates_frame_tokens_and_status(
        self, live_monitor, make_runtime_event
    ):
        _seed_session_and_turn(live_monitor, make_runtime_event)
        live_monitor.ingest_event(
            make_runtime_event(
                FrameStartEvent,
                session_id="session-1",
                turn_id="turn-1",
                frame_id="frame-1",
                iteration=0,
                context_snapshot_id="snap-1",
                context_hash="hash-1",
                timestamp=_ts(3),
            )
        )
        live_monitor.ingest_event(
            make_runtime_event(
                LLMCompleteEvent,
                session_id="session-1",
                turn_id="turn-1",
                frame_id="frame-1",
                content="the answer",
                model="gpt-4",
                provider="openai",
                finish_reason="stop",
                input_tokens=100,
                output_tokens=50,
                cached_tokens=10,
                tool_calls=[{"id": "tc-1", "name": "echo"}],
                timestamp=_ts(4),
            )
        )

        frame = live_monitor.get_session("session-1").turns["turn-1"].frames["frame-1"]
        turn = live_monitor.get_session("session-1").turns["turn-1"]
        assert frame.status == "running_tools"
        assert frame.model_name == "gpt-4"
        assert frame.input_tokens == 100
        assert frame.output_tokens == 50
        assert frame.cache_tokens == 10
        assert turn.input_tokens == 100
        assert turn.output_tokens == 50
        assert turn.cache_tokens == 10

    def test_llm_complete_falls_back_to_prompt_completion_tokens(
        self, live_monitor, make_runtime_event
    ):
        _seed_session_and_turn(live_monitor, make_runtime_event)
        live_monitor.ingest_event(
            make_runtime_event(
                FrameStartEvent,
                session_id="session-1",
                turn_id="turn-1",
                frame_id="frame-1",
                iteration=0,
                context_snapshot_id="snap-1",
                context_hash="hash-1",
                timestamp=_ts(3),
            )
        )
        live_monitor.ingest_event(
            make_runtime_event(
                LLMCompleteEvent,
                session_id="session-1",
                turn_id="turn-1",
                frame_id="frame-1",
                content="x",
                model="gpt-4",
                input_tokens=None,
                output_tokens=None,
                prompt_tokens=10,
                completion_tokens=5,
                timestamp=_ts(4),
            )
        )

        frame = live_monitor.get_session("session-1").turns["turn-1"].frames["frame-1"]
        assert frame.input_tokens == 10
        assert frame.output_tokens == 5


class TestToolEvents:
    def test_tool_start_increments_call_count(self, live_monitor, make_runtime_event):
        _seed_session_and_turn(live_monitor, make_runtime_event)
        live_monitor.ingest_event(
            make_runtime_event(
                FrameStartEvent,
                session_id="session-1",
                turn_id="turn-1",
                frame_id="frame-1",
                iteration=0,
                context_snapshot_id="snap-1",
                context_hash="hash-1",
                timestamp=_ts(3),
            )
        )
        live_monitor.ingest_event(
            make_runtime_event(
                ToolStartEvent,
                session_id="session-1",
                turn_id="turn-1",
                frame_id="frame-1",
                tool_call_id="tc-1",
                tool_name="echo",
                arguments={"text": "hi"},
                execution_mode="local",
                timestamp=_ts(4),
            )
        )

        frame = live_monitor.get_session("session-1").turns["turn-1"].frames["frame-1"]
        assert frame.status == "running_tools"
        assert frame.tool_call_count == 1

    def test_tool_result_failure_marks_frame_failed(
        self, live_monitor, make_runtime_event
    ):
        _seed_session_and_turn(live_monitor, make_runtime_event)
        live_monitor.ingest_event(
            make_runtime_event(
                FrameStartEvent,
                session_id="session-1",
                turn_id="turn-1",
                frame_id="frame-1",
                iteration=0,
                context_snapshot_id="snap-1",
                context_hash="hash-1",
                timestamp=_ts(3),
            )
        )
        live_monitor.ingest_event(
            make_runtime_event(
                ToolResultEvent,
                session_id="session-1",
                turn_id="turn-1",
                frame_id="frame-1",
                tool_call_id="tc-1",
                tool_name="echo",
                success=False,
                error={"message": "boom"},
                timestamp=_ts(4),
            )
        )

        frame = live_monitor.get_session("session-1").turns["turn-1"].frames["frame-1"]
        assert frame.status == "failed"
        assert frame.error_message == "boom"
        assert frame.finished_at is not None

    def test_tool_result_success_keeps_running_tools(
        self, live_monitor, make_runtime_event
    ):
        _seed_session_and_turn(live_monitor, make_runtime_event)
        live_monitor.ingest_event(
            make_runtime_event(
                FrameStartEvent,
                session_id="session-1",
                turn_id="turn-1",
                frame_id="frame-1",
                iteration=0,
                context_snapshot_id="snap-1",
                context_hash="hash-1",
                timestamp=_ts(3),
            )
        )
        live_monitor.ingest_event(
            make_runtime_event(
                ToolResultEvent,
                session_id="session-1",
                turn_id="turn-1",
                frame_id="frame-1",
                tool_call_id="tc-1",
                tool_name="echo",
                success=True,
                output="ok",
                timestamp=_ts(4),
            )
        )

        frame = live_monitor.get_session("session-1").turns["turn-1"].frames["frame-1"]
        assert frame.status == "running_tools"
        assert frame.error_message is None


class TestErrorEvent:
    def test_error_marks_turn_and_session_error(self, live_monitor, make_runtime_event):
        _seed_session_and_turn(live_monitor, make_runtime_event)
        live_monitor.ingest_event(
            make_runtime_event(
                ErrorEvent,
                session_id="session-1",
                turn_id="turn-1",
                error_type="runtime_error",
                error_message="something broke",
                timestamp=_ts(4),
            )
        )

        turn = live_monitor.get_session("session-1").turns["turn-1"]
        session = live_monitor.get_session("session-1")
        assert turn.status == "error"
        assert session.status == "error"


class TestRunSummaries:
    def test_get_run_summaries_sorted_by_started_at_desc(
        self, live_monitor, make_runtime_event
    ):
        live_monitor.ingest_event(
            make_runtime_event(
                SessionStartEvent,
                session_id="session-1",
                user_input="a",
                agent_mode="frame_agent",
                timestamp=_ts(1),
            )
        )
        live_monitor.ingest_event(
            make_runtime_event(
                TurnStartEvent,
                session_id="session-1",
                turn_id="turn-1",
                turn_index=0,
                user_input="a",
                timestamp=_ts(2),
            )
        )
        live_monitor.ingest_event(
            make_runtime_event(
                SessionStartEvent,
                session_id="session-2",
                user_input="b",
                agent_mode="frame_agent",
                timestamp=_ts(5),
            )
        )
        live_monitor.ingest_event(
            make_runtime_event(
                TurnStartEvent,
                session_id="session-2",
                turn_id="turn-2",
                turn_index=0,
                user_input="b",
                timestamp=_ts(6),
            )
        )

        summaries = live_monitor.get_run_summaries()
        assert [s.run_id for s in summaries] == ["turn-2", "turn-1"]
        assert summaries[0].session_id == "session-2"
        assert summaries[1].session_id == "session-1"

    def test_get_run_summaries_current_frame_assertions(
        self, live_monitor, make_runtime_event
    ):
        _seed_session_turn_frame(live_monitor, make_runtime_event)

        summaries = live_monitor.get_run_summaries()
        assert summaries[0].current_frame_id == "frame-1"
        assert summaries[0].current_frame_status == "running_llm"

    def test_get_run_summaries_turn_without_frames(
        self, live_monitor, make_runtime_event
    ):
        live_monitor.ingest_event(
            make_runtime_event(
                SessionStartEvent,
                session_id="session-1",
                user_input="a",
                agent_mode="frame_agent",
                timestamp=_ts(1),
            )
        )
        live_monitor.ingest_event(
            make_runtime_event(
                TurnStartEvent,
                session_id="session-1",
                turn_id="turn-1",
                turn_index=0,
                user_input="a",
                timestamp=_ts(2),
            )
        )

        summaries = live_monitor.get_run_summaries()
        assert summaries[0].current_frame_id is None
        assert summaries[0].current_frame_status is None


class TestActiveSessions:
    def test_get_active_sessions_filters_running_paused_interrupted(
        self, live_monitor, make_runtime_event
    ):
        live_monitor.ingest_event(
            make_runtime_event(
                SessionStartEvent,
                session_id="session-1",
                user_input="a",
                agent_mode="frame_agent",
                timestamp=_ts(1),
            )
        )
        live_monitor.ingest_event(
            make_runtime_event(
                SessionStartEvent,
                session_id="session-2",
                user_input="b",
                agent_mode="frame_agent",
                timestamp=_ts(2),
            )
        )
        live_monitor.ingest_event(
            make_runtime_event(
                SessionInterruptEvent,
                session_id="session-2",
                turn_id="turn-2",
                timestamp=_ts(3),
            )
        )

        active = live_monitor.get_active_sessions()
        assert [s.session_id for s in active] == ["session-2", "session-1"]
        assert active[0].status == "interrupted"
        assert active[1].status == "running"

    def test_get_session_returns_none_for_missing(self, live_monitor):
        assert live_monitor.get_session("nope") is None


class TestTurnEvents:
    def test_get_turn_events_sorted_by_seq(self, live_monitor, make_runtime_event):
        live_monitor.ingest_event(
            make_runtime_event(
                SessionStartEvent,
                session_id="session-1",
                user_input="a",
                agent_mode="frame_agent",
                seq=1,
                timestamp=_ts(1),
            )
        )
        live_monitor.ingest_event(
            make_runtime_event(
                TurnStartEvent,
                session_id="session-1",
                turn_id="turn-1",
                turn_index=0,
                user_input="a",
                seq=3,
                timestamp=_ts(2),
            )
        )
        live_monitor.ingest_event(
            make_runtime_event(
                FrameStartEvent,
                session_id="session-1",
                turn_id="turn-1",
                frame_id="frame-1",
                iteration=0,
                context_snapshot_id="snap-1",
                context_hash="hash-1",
                seq=2,
                timestamp=_ts(3),
            )
        )

        events = live_monitor.get_turn_events("turn-1")
        assert [e.seq for e in events] == [1, 2, 3]
        assert [e.type for e in events] == [
            EventType.SESSION_START,
            EventType.FRAME_START,
            EventType.TURN_START,
        ]

    def test_get_turn_events_empty_for_unknown_turn(self, live_monitor):
        assert live_monitor.get_turn_events("nope") == []


class TestStats:
    def test_get_stats(self, live_monitor, make_runtime_event):
        live_monitor.ingest_event(
            make_runtime_event(
                SessionStartEvent,
                session_id="session-1",
                user_input="a",
                agent_mode="frame_agent",
                timestamp=_ts(1),
            )
        )
        live_monitor.ingest_event(
            make_runtime_event(
                TurnStartEvent,
                session_id="session-1",
                turn_id="turn-1",
                turn_index=0,
                user_input="a",
                timestamp=_ts(2),
            )
        )

        stats = live_monitor.get_stats()
        assert stats.active_sessions == 1
        assert stats.running_turns == 1
        assert stats.total_events == 2
        assert stats.events_per_second > 0

    def test_get_stats_counts_recent_errors(self, live_monitor, make_runtime_event):
        live_monitor.ingest_event(
            make_runtime_event(
                SessionStartEvent,
                session_id="session-1",
                user_input="a",
                agent_mode="frame_agent",
                timestamp=_ts(1),
            )
        )
        live_monitor.ingest_event(
            make_runtime_event(
                ErrorEvent,
                session_id="session-1",
                turn_id="turn-1",
                error_type="e",
                error_message="m",
                timestamp=_ts(3),
            )
        )

        stats = live_monitor.get_stats()
        assert stats.error_count_last_minute == 1

    def test_clear_resets_state(self, live_monitor, make_runtime_event):
        live_monitor.ingest_event(
            make_runtime_event(
                SessionStartEvent,
                session_id="session-1",
                user_input="a",
                agent_mode="frame_agent",
                timestamp=_ts(1),
            )
        )
        assert live_monitor.get_session("session-1") is not None
        live_monitor.clear()
        assert live_monitor.get_session("session-1") is None
        assert live_monitor.get_stats().total_events == 0

    def test_max_events_per_frame_truncation(self, make_runtime_event):
        monitor = LiveMonitor(max_events_per_frame=2)
        _seed_session_turn_frame(monitor, make_runtime_event)
        for i in range(5):
            monitor.ingest_event(
                make_runtime_event(
                    LLMChunkEvent,
                    session_id="session-1",
                    turn_id="turn-1",
                    frame_id="frame-1",
                    seq=i,
                    content=f"chunk {i}",
                    timestamp=_ts(i),
                )
            )

        frame = monitor.get_session("session-1").turns["turn-1"].frames["frame-1"]
        assert len(frame.events) == 2
        assert [e.seq for e in frame.events] == [3, 4]
