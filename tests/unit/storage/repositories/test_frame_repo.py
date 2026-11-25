"""Frame repository unit tests."""

from __future__ import annotations

import uuid


class TestFrameRepo:
    async def test_create_inserts_frame_with_all_columns(
        self, frame_repo, session_repo, turn_repo
    ) -> None:
        """create(): inserts frame with iteration, context_snapshot_id, history_through_seq, prompt_hash."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        turn_id = str(uuid.uuid4())
        await turn_repo.create(
            turn_id=turn_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            turn_index=0,
            user_input="turn query",
            status="running",
            started_at_ms=1000,
            config_snapshot_json="{}",
        )

        frame_id = str(uuid.uuid4())
        context_snapshot_id = str(uuid.uuid4())
        prompt_hash = "abc123"
        await frame_repo.create(
            frame_id=frame_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            iteration=1,
            status="running_llm",
            started_at_ms=1000,
            model_name="gpt-4",
            turn_id=turn_id,
            context_snapshot_id=context_snapshot_id,
            history_through_seq=5,
            prompt_hash=prompt_hash,
        )

        frame = await frame_repo.get(frame_id)
        assert frame is not None
        assert frame.id == frame_id
        assert frame.session_id == session_id
        assert frame.turn_id == turn_id
        assert frame.iteration == 1
        assert frame.status == "running_llm"
        assert frame.started_at_ms == 1000
        assert frame.model_name == "gpt-4"
        assert frame.context_snapshot_id == context_snapshot_id
        assert frame.history_through_seq == 5
        assert frame.prompt_hash == prompt_hash
        assert frame.finished_at_ms is None
        assert frame.duration_ms is None
        assert frame.input_tokens == 0
        assert frame.output_tokens == 0
        assert frame.finish_reason is None
        assert frame.error_message is None
        assert frame.pending_call_id is None

    async def test_get_returns_stored_frame_with_all_optional_fields(
        self, frame_repo, session_repo, turn_repo
    ) -> None:
        """get(): returns StoredFrame with all optional fields."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        turn_id = str(uuid.uuid4())
        await turn_repo.create(
            turn_id=turn_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            turn_index=0,
            user_input="turn query",
            status="running",
            started_at_ms=1000,
            config_snapshot_json="{}",
        )

        frame_id = str(uuid.uuid4())
        await frame_repo.create(
            frame_id=frame_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            iteration=1,
            status="running_llm",
            started_at_ms=1000,
            turn_id=turn_id,
        )

        # Update with all optional fields
        await frame_repo.update_status(
            frame_id=frame_id,
            status="completed",
            finished_at_ms=5000,
            duration_ms=4000,
            input_tokens=100,
            output_tokens=0,
            finish_reason="stop",
            error_message=None,
            pending_call_id=None,
        )

        frame = await frame_repo.get(frame_id)
        assert frame is not None
        assert frame.finished_at_ms == 5000
        assert frame.duration_ms == 4000
        assert frame.input_tokens == 100
        assert frame.output_tokens == 0
        assert frame.finish_reason == "stop"

    async def test_get_returns_none_for_missing_frame(self, frame_repo) -> None:
        frame = await frame_repo.get(str(uuid.uuid4()))
        assert frame is None

    async def test_update_status_coalesce_tokens_used(
        self, frame_repo, session_repo, turn_repo
    ) -> None:
        """update_status(): finished_at, duration_ms, tokens_used (COALESCE), finish_reason, error_message, pending_call_id."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        turn_id = str(uuid.uuid4())
        await turn_repo.create(
            turn_id=turn_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            turn_index=0,
            user_input="turn query",
            status="running",
            started_at_ms=1000,
            config_snapshot_json="{}",
        )

        frame_id = str(uuid.uuid4())
        await frame_repo.create(
            frame_id=frame_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            iteration=1,
            status="running_llm",
            started_at_ms=1000,
            turn_id=turn_id,
        )

        # Set initial tokens
        await frame_repo.update_status(
            frame_id=frame_id, status="running_tools", input_tokens=50, output_tokens=0
        )

        # Update without input_tokens (should preserve via COALESCE)
        await frame_repo.update_status(
            frame_id=frame_id,
            status="completed",
            finished_at_ms=5000,
            finish_reason="stop",
        )

        frame = await frame_repo.get(frame_id)
        assert frame is not None
        assert frame.input_tokens == 50  # preserved via COALESCE
        assert frame.output_tokens == 0
        assert frame.finished_at_ms == 5000
        assert frame.finish_reason == "stop"

    async def test_update_status_sets_error_message_and_pending_call_id(
        self, frame_repo, session_repo, turn_repo
    ) -> None:
        """update_status(): sets error_message and pending_call_id."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        turn_id = str(uuid.uuid4())
        await turn_repo.create(
            turn_id=turn_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            turn_index=0,
            user_input="turn query",
            status="running",
            started_at_ms=1000,
            config_snapshot_json="{}",
        )

        frame_id = str(uuid.uuid4())
        await frame_repo.create(
            frame_id=frame_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            iteration=1,
            status="running_llm",
            started_at_ms=1000,
            turn_id=turn_id,
        )

        await frame_repo.update_status(
            frame_id=frame_id,
            status="paused_approval",
            error_message="Tool requires approval",
            pending_call_id="call_123",
        )

        frame = await frame_repo.get(frame_id)
        assert frame is not None
        assert frame.status == "paused_approval"
        assert frame.error_message == "Tool requires approval"
        assert frame.pending_call_id == "call_123"

    async def test_increment_tokens_atomic_addition(
        self, frame_repo, session_repo, turn_repo
    ) -> None:
        """increment_tokens(): atomic token addition."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        turn_id = str(uuid.uuid4())
        await turn_repo.create(
            turn_id=turn_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            turn_index=0,
            user_input="turn query",
            status="running",
            started_at_ms=1000,
            config_snapshot_json="{}",
        )

        frame_id = str(uuid.uuid4())
        await frame_repo.create(
            frame_id=frame_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            iteration=1,
            status="running_llm",
            started_at_ms=1000,
            turn_id=turn_id,
        )

        # Set initial tokens
        await frame_repo.update_status(
            frame_id=frame_id, status="running_tools", input_tokens=50, output_tokens=0
        )

        # Increment
        await frame_repo.increment_tokens(frame_id, input_tokens=25, output_tokens=0)
        frame = await frame_repo.get(frame_id)
        assert frame is not None
        assert frame.input_tokens == 75
        assert frame.output_tokens == 0

    async def test_increment_tokens_from_null(
        self, frame_repo, session_repo, turn_repo
    ) -> None:
        """increment_tokens() handles NULL initial values via COALESCE."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        turn_id = str(uuid.uuid4())
        await turn_repo.create(
            turn_id=turn_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            turn_index=0,
            user_input="turn query",
            status="running",
            started_at_ms=1000,
            config_snapshot_json="{}",
        )

        frame_id = str(uuid.uuid4())
        await frame_repo.create(
            frame_id=frame_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            iteration=1,
            status="running_llm",
            started_at_ms=1000,
            turn_id=turn_id,
        )

        # Don't set initial tokens
        await frame_repo.increment_tokens(frame_id, input_tokens=25, output_tokens=0)
        frame = await frame_repo.get(frame_id)
        assert frame is not None
        assert frame.input_tokens == 25
        assert frame.output_tokens == 0

    async def test_list_by_session_ordered_by_iteration(
        self, frame_repo, session_repo, turn_repo
    ) -> None:
        """list_by_session(): ordered by iteration."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        turn_id = str(uuid.uuid4())
        await turn_repo.create(
            turn_id=turn_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            turn_index=0,
            user_input="turn query",
            status="running",
            started_at_ms=1000,
            config_snapshot_json="{}",
        )

        frame1_id = str(uuid.uuid4())
        frame2_id = str(uuid.uuid4())
        frame3_id = str(uuid.uuid4())

        # Create frames out of order
        await frame_repo.create(
            frame_id=frame2_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            iteration=2,
            status="running_llm",
            started_at_ms=1000,
            turn_id=turn_id,
        )
        await frame_repo.create(
            frame_id=frame1_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            iteration=0,
            status="running_llm",
            started_at_ms=1000,
            turn_id=turn_id,
        )
        await frame_repo.create(
            frame_id=frame3_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            iteration=1,
            status="running_llm",
            started_at_ms=1000,
            turn_id=turn_id,
        )

        frames = await frame_repo.list_by_session(session_id)
        assert len(frames) == 3
        assert frames[0].id == frame1_id  # iteration=0
        assert frames[1].id == frame3_id  # iteration=1
        assert frames[2].id == frame2_id  # iteration=2

    async def test_list_by_turn_ordered_by_iteration(
        self, frame_repo, session_repo, turn_repo
    ) -> None:
        """list_by_turn(): ordered by iteration."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        turn_id = str(uuid.uuid4())
        await turn_repo.create(
            turn_id=turn_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            turn_index=0,
            user_input="turn query",
            status="running",
            started_at_ms=1000,
            config_snapshot_json="{}",
        )

        frame1_id = str(uuid.uuid4())
        frame2_id = str(uuid.uuid4())
        frame3_id = str(uuid.uuid4())

        await frame_repo.create(
            frame_id=frame2_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            iteration=2,
            status="running_llm",
            started_at_ms=1000,
            turn_id=turn_id,
        )
        await frame_repo.create(
            frame_id=frame1_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            iteration=0,
            status="running_llm",
            started_at_ms=1000,
            turn_id=turn_id,
        )
        await frame_repo.create(
            frame_id=frame3_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            iteration=1,
            status="running_llm",
            started_at_ms=1000,
            turn_id=turn_id,
        )

        frames = await frame_repo.list_by_turn(turn_id)
        assert len(frames) == 3
        assert frames[0].id == frame1_id  # iteration=0
        assert frames[1].id == frame3_id  # iteration=1
        assert frames[2].id == frame2_id  # iteration=2

    async def test_list_ids_by_turn_returns_only_ids(
        self, frame_repo, session_repo, turn_repo
    ) -> None:
        """list_ids_by_turn(): returns only frame IDs."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        turn_id = str(uuid.uuid4())
        await turn_repo.create(
            turn_id=turn_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            turn_index=0,
            user_input="turn query",
            status="running",
            started_at_ms=1000,
            config_snapshot_json="{}",
        )

        frame1_id = str(uuid.uuid4())
        frame2_id = str(uuid.uuid4())

        await frame_repo.create(
            frame_id=frame1_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            iteration=0,
            status="running_llm",
            started_at_ms=1000,
            turn_id=turn_id,
        )
        await frame_repo.create(
            frame_id=frame2_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            iteration=1,
            status="running_llm",
            started_at_ms=1000,
            turn_id=turn_id,
        )

        frame_ids = await frame_repo.list_ids_by_turn(turn_id)
        assert len(frame_ids) == 2
        assert frame1_id in frame_ids
        assert frame2_id in frame_ids
        assert all(isinstance(id, str) for id in frame_ids)

    async def test_get_running_frames_filters_status(
        self, frame_repo, session_repo, turn_repo
    ) -> None:
        """get_running_frames(): filters status IN ('running_llm', 'running_tools')."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        turn_id = str(uuid.uuid4())
        await turn_repo.create(
            turn_id=turn_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            turn_index=0,
            user_input="turn query",
            status="running",
            started_at_ms=1000,
            config_snapshot_json="{}",
        )

        frame1_id = str(uuid.uuid4())
        frame2_id = str(uuid.uuid4())
        frame3_id = str(uuid.uuid4())

        await frame_repo.create(
            frame_id=frame1_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            iteration=0,
            status="running_llm",
            started_at_ms=1000,
            turn_id=turn_id,
        )
        await frame_repo.create(
            frame_id=frame2_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            iteration=1,
            status="running_tools",
            started_at_ms=1000,
            turn_id=turn_id,
        )
        await frame_repo.create(
            frame_id=frame3_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            iteration=2,
            status="completed",
            started_at_ms=1000,
            turn_id=turn_id,
        )

        running_frames = await frame_repo.get_running_frames()
        assert len(running_frames) == 2
        running_frame_ids = {f.id for f in running_frames}
        assert frame1_id in running_frame_ids
        assert frame2_id in running_frame_ids
        assert frame3_id not in running_frame_ids

    async def test_mark_failed_sets_failed_status_with_error(
        self, frame_repo, session_repo, turn_repo
    ) -> None:
        """mark_failed(): sets failed status with error message."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        turn_id = str(uuid.uuid4())
        await turn_repo.create(
            turn_id=turn_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            turn_index=0,
            user_input="turn query",
            status="running",
            started_at_ms=1000,
            config_snapshot_json="{}",
        )

        frame_id = str(uuid.uuid4())
        await frame_repo.create(
            frame_id=frame_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            iteration=1,
            status="running_llm",
            started_at_ms=1000,
            turn_id=turn_id,
        )

        await frame_repo.mark_failed(frame_id, error_message="Crash detected")

        frame = await frame_repo.get(frame_id)
        assert frame is not None
        assert frame.status == "failed"
        assert frame.error_message == "Crash detected"
        assert frame.finished_at_ms is not None  # Set by mark_failed

    async def test_upsert_creates_new_frame(
        self, frame_repo, session_repo, turn_repo
    ) -> None:
        """upsert(): creates new frame when ID doesn't exist."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        turn_id = str(uuid.uuid4())
        await turn_repo.create(
            turn_id=turn_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            turn_index=0,
            user_input="turn query",
            status="running",
            started_at_ms=1000,
            config_snapshot_json="{}",
        )

        frame_id = str(uuid.uuid4())
        await frame_repo.upsert(
            frame_id=frame_id,
            session_id=session_id,
            turn_id=turn_id,
            trace_id=str(uuid.uuid4()),
            iteration=1,
            status="running_llm",
            started_at_ms=1000,
        )

        frame = await frame_repo.get(frame_id)
        assert frame is not None
        assert frame.id == frame_id
        assert frame.status == "running_llm"

    async def test_upsert_tokens_used_null_keeps_existing(
        self, frame_repo, session_repo, turn_repo
    ) -> None:
        """upsert(): tokens_used: NULL = keep existing, else overwrite."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        turn_id = str(uuid.uuid4())
        await turn_repo.create(
            turn_id=turn_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            turn_index=0,
            user_input="turn query",
            status="running",
            started_at_ms=1000,
            config_snapshot_json="{}",
        )

        frame_id = str(uuid.uuid4())
        await frame_repo.create(
            frame_id=frame_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            iteration=1,
            status="running_llm",
            started_at_ms=1000,
            turn_id=turn_id,
        )
        await frame_repo.update_status(
            frame_id=frame_id, status="running_tools", input_tokens=50, output_tokens=0
        )

        # Upsert with NULL input_tokens should preserve existing
        await frame_repo.upsert(
            frame_id=frame_id,
            session_id=session_id,
            turn_id=turn_id,
            trace_id=str(uuid.uuid4()),
            iteration=1,
            status="completed",
            started_at_ms=1000,
            finished_at_ms=5000,
            input_tokens=None,  # NULL = keep existing
            output_tokens=None,
        )

        frame = await frame_repo.get(frame_id)
        assert frame is not None
        assert frame.input_tokens == 50  # preserved
        assert frame.output_tokens == 0

    async def test_upsert_tokens_used_value_overwrites(
        self, frame_repo, session_repo, turn_repo
    ) -> None:
        """upsert(): tokens_used with value overwrites existing."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        turn_id = str(uuid.uuid4())
        await turn_repo.create(
            turn_id=turn_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            turn_index=0,
            user_input="turn query",
            status="running",
            started_at_ms=1000,
            config_snapshot_json="{}",
        )

        frame_id = str(uuid.uuid4())
        await frame_repo.create(
            frame_id=frame_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            iteration=1,
            status="running_llm",
            started_at_ms=1000,
            turn_id=turn_id,
        )
        await frame_repo.update_status(
            frame_id=frame_id, status="running_tools", input_tokens=50, output_tokens=0
        )

        # Upsert with value should overwrite
        await frame_repo.upsert(
            frame_id=frame_id,
            session_id=session_id,
            turn_id=turn_id,
            trace_id=str(uuid.uuid4()),
            iteration=1,
            status="completed",
            started_at_ms=1000,
            finished_at_ms=5000,
            input_tokens=100,  # overwrites
            output_tokens=0,
        )

        frame = await frame_repo.get(frame_id)
        assert frame is not None
        assert frame.input_tokens == 100  # overwritten
        assert frame.output_tokens == 0

    async def test_upsert_history_through_seq_zero_keeps_existing(
        self, frame_repo, session_repo, turn_repo
    ) -> None:
        """upsert(): history_through_seq: 0 = keep existing, else overwrite."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        turn_id = str(uuid.uuid4())
        await turn_repo.create(
            turn_id=turn_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            turn_index=0,
            user_input="turn query",
            status="running",
            started_at_ms=1000,
            config_snapshot_json="{}",
        )

        frame_id = str(uuid.uuid4())
        await frame_repo.create(
            frame_id=frame_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            iteration=1,
            status="running_llm",
            started_at_ms=1000,
            turn_id=turn_id,
            history_through_seq=5,
        )

        # Upsert with history_through_seq=0 should preserve existing
        await frame_repo.upsert(
            frame_id=frame_id,
            session_id=session_id,
            turn_id=turn_id,
            trace_id=str(uuid.uuid4()),
            iteration=1,
            status="completed",
            started_at_ms=1000,
            history_through_seq=0,  # 0 = keep existing
        )

        frame = await frame_repo.get(frame_id)
        assert frame is not None
        assert frame.history_through_seq == 5  # preserved

    async def test_upsert_history_through_seq_value_overwrites(
        self, frame_repo, session_repo, turn_repo
    ) -> None:
        """upsert(): history_through_seq with value overwrites existing."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        turn_id = str(uuid.uuid4())
        await turn_repo.create(
            turn_id=turn_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            turn_index=0,
            user_input="turn query",
            status="running",
            started_at_ms=1000,
            config_snapshot_json="{}",
        )

        frame_id = str(uuid.uuid4())
        await frame_repo.create(
            frame_id=frame_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            iteration=1,
            status="running_llm",
            started_at_ms=1000,
            turn_id=turn_id,
            history_through_seq=5,
        )

        # Upsert with value should overwrite
        await frame_repo.upsert(
            frame_id=frame_id,
            session_id=session_id,
            turn_id=turn_id,
            trace_id=str(uuid.uuid4()),
            iteration=1,
            status="completed",
            started_at_ms=1000,
            history_through_seq=10,  # overwrites
        )

        frame = await frame_repo.get(frame_id)
        assert frame is not None
        assert frame.history_through_seq == 10  # overwritten

    async def test_upsert_coalesce_prompt_hash(
        self, frame_repo, session_repo, turn_repo
    ) -> None:
        """upsert(): prompt_hash: COALESCE (update only if provided)."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        turn_id = str(uuid.uuid4())
        await turn_repo.create(
            turn_id=turn_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            turn_index=0,
            user_input="turn query",
            status="running",
            started_at_ms=1000,
            config_snapshot_json="{}",
        )

        frame_id = str(uuid.uuid4())
        await frame_repo.create(
            frame_id=frame_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            iteration=1,
            status="running_llm",
            started_at_ms=1000,
            turn_id=turn_id,
            prompt_hash="abc123",
        )

        # Upsert with NULL prompt_hash should preserve existing via COALESCE
        await frame_repo.upsert(
            frame_id=frame_id,
            session_id=session_id,
            turn_id=turn_id,
            trace_id=str(uuid.uuid4()),
            iteration=1,
            status="completed",
            started_at_ms=1000,
            prompt_hash=None,  # NULL = keep existing via COALESCE
        )

        frame = await frame_repo.get(frame_id)
        assert frame is not None
        assert frame.prompt_hash == "abc123"  # preserved

        # Upsert with value should overwrite
        await frame_repo.upsert(
            frame_id=frame_id,
            session_id=session_id,
            turn_id=turn_id,
            trace_id=str(uuid.uuid4()),
            iteration=1,
            status="completed",
            started_at_ms=1000,
            prompt_hash="def456",  # overwrites
        )

        frame = await frame_repo.get(frame_id)
        assert frame is not None
        assert frame.prompt_hash == "def456"  # overwritten

    async def test_upsert_partial_update_with_defaults_preserves_insert_only_fields(
        self, frame_repo, session_repo, turn_repo
    ) -> None:
        """upsert(): omitting iteration/started_at_ms on update preserves them via defaults; COALESCE preserves finished_at/duration_ms."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        turn_id = str(uuid.uuid4())
        await turn_repo.create(
            turn_id=turn_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            turn_index=0,
            user_input="turn query",
            status="running",
            started_at_ms=1000,
            config_snapshot_json="{}",
        )

        frame_id = str(uuid.uuid4())
        await frame_repo.create(
            frame_id=frame_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            iteration=1,
            status="running_llm",
            started_at_ms=1000,
            turn_id=turn_id,
        )

        # Upsert with only status + model_name (defaults for iteration/started_at_ms/finished_at_ms/duration_ms)
        await frame_repo.upsert(
            frame_id=frame_id,
            session_id=session_id,
            turn_id=turn_id,
            trace_id=str(uuid.uuid4()),
            status="running_tools",
            model_name="gpt-4",
        )

        frame = await frame_repo.get(frame_id)
        assert frame is not None
        assert frame.status == "running_tools"
        assert frame.model_name == "gpt-4"
        # INSERT-only fields preserved (not in UPDATE SET; defaults satisfy signature)
        assert frame.iteration == 1
        assert frame.started_at_ms == 1000
        # finished_at/duration_ms preserved via COALESCE when None passed
        assert frame.finished_at_ms is None
        assert frame.duration_ms is None

        # Finish the frame
        await frame_repo.upsert(
            frame_id=frame_id,
            session_id=session_id,
            turn_id=turn_id,
            trace_id=str(uuid.uuid4()),
            status="completed",
            finished_at_ms=5000,
            duration_ms=4000,
        )
        frame = await frame_repo.get(frame_id)
        assert frame.finished_at_ms == 5000
        assert frame.duration_ms == 4000

        # Subsequent update without finished_at_ms/duration_ms preserves them via COALESCE
        await frame_repo.upsert(
            frame_id=frame_id,
            session_id=session_id,
            turn_id=turn_id,
            trace_id=str(uuid.uuid4()),
            status="completed",
        )
        frame = await frame_repo.get(frame_id)
        assert frame.finished_at_ms == 5000
        assert frame.duration_ms == 4000
