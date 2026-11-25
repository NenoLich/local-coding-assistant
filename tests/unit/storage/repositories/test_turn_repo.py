"""Turn repository unit tests."""

from __future__ import annotations

import uuid


class TestTurnRepo:
    async def test_create_inserts_turn_with_all_columns(
        self, turn_repo, session_repo
    ) -> None:
        """create(): inserts turn with config_snapshot_json, context_snapshot_id."""
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
        context_snapshot_id = str(uuid.uuid4())
        await turn_repo.create(
            turn_id=turn_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            turn_index=0,
            user_input="turn query",
            status="running",
            started_at_ms=1000,
            context_snapshot_id=context_snapshot_id,
            config_snapshot_json='{"context": "config"}',
        )

        turn = await turn_repo.get(turn_id)
        assert turn is not None
        assert turn.id == turn_id
        assert turn.session_id == session_id
        assert turn.turn_index == 0
        assert turn.user_input == "turn query"
        assert turn.status == "running"
        assert turn.started_at_ms == 1000
        assert turn.context_snapshot_id == context_snapshot_id
        assert turn.config_snapshot_json == '{"context": "config"}'
        assert turn.finished_at_ms is None
        assert turn.paused_at_ms is None
        assert turn.pause_reason is None
        assert turn.resumed_at_ms is None
        assert turn.total_input_tokens == 0
        assert turn.total_output_tokens == 0
        assert turn.total_frames == 0
        assert turn.final_answer is None

    async def test_get_returns_stored_turn_with_pause_resume_fields(
        self, turn_repo, session_repo
    ) -> None:
        """get(): returns StoredTurn with pause/resume fields."""
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

        # Update with pause/resume fields
        await turn_repo.update_status(
            turn_id=turn_id,
            status="paused",
            paused_at_ms=2000,
            pause_reason="approval",
            resumed_at_ms=3000,
        )

        turn = await turn_repo.get(turn_id)
        assert turn is not None
        assert turn.paused_at_ms == 2000
        assert turn.pause_reason == "approval"
        assert turn.resumed_at_ms == 3000

    async def test_get_returns_none_for_missing_turn(self, turn_repo) -> None:
        turn = await turn_repo.get(str(uuid.uuid4()))
        assert turn is None

    async def test_update_status_handles_finished_at_and_totals(
        self, turn_repo, session_repo
    ) -> None:
        """update_status(): handles finished_at, paused_at, pause_reason, resumed_at, totals."""
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

        await turn_repo.update_status(
            turn_id=turn_id,
            status="completed",
            finished_at_ms=5000,
            total_input_tokens=60,
            total_output_tokens=40,
            total_frames=5,
        )

        turn = await turn_repo.get(turn_id)
        assert turn is not None
        assert turn.status == "completed"
        assert turn.finished_at_ms == 5000
        assert turn.total_input_tokens == 60
        assert turn.total_output_tokens == 40
        assert turn.total_frames == 5

    async def test_update_status_final_answer_persisted(
        self, turn_repo, session_repo
    ) -> None:
        """update_status(): final_answer is persisted."""
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

        await turn_repo.update_status(
            turn_id=turn_id,
            status="completed",
            finished_at_ms=5000,
            final_answer="The answer is 42",
        )

        turn = await turn_repo.get(turn_id)
        assert turn is not None
        assert turn.status == "completed"
        assert turn.final_answer == "The answer is 42"

    async def test_update_status_coalesce_preserves_existing(
        self, turn_repo, session_repo
    ) -> None:
        """update_status(): COALESCE preserves existing paused_at/pause_reason/resumed_at when None passed."""
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

        # Set pause fields
        await turn_repo.update_status(
            turn_id=turn_id,
            status="paused",
            paused_at_ms=2000,
            pause_reason="approval",
        )

        # Update status without pause fields (should preserve via COALESCE)
        await turn_repo.update_status(
            turn_id=turn_id,
            status="running",
        )

        turn = await turn_repo.get(turn_id)
        assert turn is not None
        assert turn.paused_at_ms == 2000  # preserved
        assert turn.pause_reason == "approval"  # preserved

    async def test_increment_totals_atomic_token_frame_deltas(
        self, turn_repo, session_repo
    ) -> None:
        """increment_totals(): atomic token/frame deltas."""
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

        # Set initial values via update_status
        await turn_repo.update_status(
            turn_id=turn_id,
            status="running",
            total_input_tokens=6,
            total_output_tokens=4,
            total_frames=2,
        )

        # Increment
        await turn_repo.increment_totals(
            turn_id, total_input_tokens=3, total_output_tokens=2, total_frames=1
        )
        turn = await turn_repo.get(turn_id)
        assert turn is not None
        assert turn.total_input_tokens == 9
        assert turn.total_output_tokens == 6
        assert turn.total_frames == 3

    async def test_increment_totals_from_null(self, turn_repo, session_repo) -> None:
        """increment_totals() handles NULL initial values via COALESCE."""
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

        # Don't set initial values
        await turn_repo.increment_totals(
            turn_id, total_input_tokens=3, total_output_tokens=2, total_frames=1
        )
        turn = await turn_repo.get(turn_id)
        assert turn is not None
        assert turn.total_input_tokens == 3
        assert turn.total_output_tokens == 2
        assert turn.total_frames == 1

    async def test_list_by_session_ordered_by_turn_index(
        self, turn_repo, session_repo
    ) -> None:
        """list_by_session(): ordered by turn_index."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        turn1_id = str(uuid.uuid4())
        turn2_id = str(uuid.uuid4())
        turn3_id = str(uuid.uuid4())

        # Create turns out of order
        await turn_repo.create(
            turn_id=turn2_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            turn_index=2,
            user_input="query2",
            status="running",
            started_at_ms=1000,
            config_snapshot_json="{}",
        )
        await turn_repo.create(
            turn_id=turn1_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            turn_index=0,
            user_input="query1",
            status="running",
            started_at_ms=1000,
            config_snapshot_json="{}",
        )
        await turn_repo.create(
            turn_id=turn3_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            turn_index=1,
            user_input="query3",
            status="running",
            started_at_ms=1000,
            config_snapshot_json="{}",
        )

        turns = await turn_repo.list_by_session(session_id)
        assert len(turns) == 3
        assert turns[0].id == turn1_id  # turn_index=0
        assert turns[1].id == turn3_id  # turn_index=1
        assert turns[2].id == turn2_id  # turn_index=2

    async def test_upsert_creates_new_turn(self, turn_repo, session_repo) -> None:
        """upsert(): creates new turn when ID doesn't exist."""
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
        await turn_repo.upsert(
            turn_id=turn_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            turn_index=0,
            user_input="query",
            status="running",
            started_at_ms=1000,
            config_snapshot_json="{}",
        )

        turn = await turn_repo.get(turn_id)
        assert turn is not None
        assert turn.id == turn_id
        assert turn.status == "running"

    async def test_upsert_partial_update_semantics(
        self, turn_repo, session_repo
    ) -> None:
        """upsert(): partial update semantics for status/finished_at/paused_at/pause_reason/resumed_at/context_snapshot_id/config_snapshot_json."""
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
            user_input="query",
            status="running",
            started_at_ms=1000,
            context_snapshot_id="snap1",
            config_snapshot_json='{"old": "config"}',
        )

        # Update with new values
        await turn_repo.upsert(
            turn_id=turn_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            turn_index=0,
            user_input="query",
            status="completed",
            started_at_ms=1000,
            finished_at_ms=5000,
            paused_at_ms=2000,
            pause_reason="approval",
            resumed_at_ms=3000,
            context_snapshot_id="snap2",
            config_snapshot_json='{"new": "config"}',
        )

        turn = await turn_repo.get(turn_id)
        assert turn is not None
        assert turn.status == "completed"
        assert turn.finished_at_ms == 5000
        assert turn.paused_at_ms == 2000
        assert turn.pause_reason == "approval"
        assert turn.resumed_at_ms == 3000
        assert turn.context_snapshot_id == "snap2"
        assert turn.config_snapshot_json == '{"new": "config"}'

    async def test_upsert_total_null_is_noop(self, turn_repo, session_repo) -> None:
        """upsert(): total_tokens/total_frames NULL = no-op (preserves existing values)."""
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
            user_input="query",
            status="running",
            started_at_ms=1000,
            config_snapshot_json="{}",
        )
        await turn_repo.update_status(
            turn_id=turn_id,
            status="running",
            total_input_tokens=60,
            total_output_tokens=40,
            total_frames=5,
        )

        await turn_repo.upsert(
            turn_id=turn_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            turn_index=0,
            user_input="query",
            status="running",
            started_at_ms=1000,
            config_snapshot_json="{}",
            total_input_tokens=None,  # NULL = no-op
            total_output_tokens=None,  # NULL = no-op
            total_frames=None,  # NULL = no-op
        )

        turn = await turn_repo.get(turn_id)
        assert turn is not None
        assert turn.total_input_tokens == 60  # preserved
        assert turn.total_output_tokens == 40  # preserved
        assert turn.total_frames == 5  # preserved

    async def test_upsert_partial_update_with_defaults_preserves_insert_only_fields(
        self, turn_repo, session_repo
    ) -> None:
        """upsert(): omitting turn_index/user_input/started_at_ms on update preserves them via defaults."""
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

        # Upsert with only status + finished_at_ms (defaults for turn_index/user_input/started_at_ms)
        await turn_repo.upsert(
            turn_id=turn_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            status="completed",
            finished_at_ms=5000,
        )

        turn = await turn_repo.get(turn_id)
        assert turn is not None
        assert turn.status == "completed"
        assert turn.finished_at_ms == 5000
        # INSERT-only fields preserved (not in UPDATE SET; defaults satisfy signature)
        assert turn.turn_index == 0
        assert turn.user_input == "turn query"
        assert turn.started_at_ms == 1000
        # finished_at preserved via COALESCE when None passed on a subsequent update
        await turn_repo.upsert(
            turn_id=turn_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            status="completed",
        )
        turn = await turn_repo.get(turn_id)
        assert turn.finished_at_ms == 5000  # preserved via COALESCE

    async def test_upsert_final_answer_persisted_and_coalesced(
        self, turn_repo, session_repo
    ) -> None:
        """upsert(): final_answer is persisted and preserved via COALESCE on null."""
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
            user_input="query",
            status="running",
            started_at_ms=1000,
            config_snapshot_json="{}",
        )

        # Set final answer via upsert
        await turn_repo.upsert(
            turn_id=turn_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            turn_index=0,
            user_input="query",
            status="completed",
            started_at_ms=1000,
            finished_at_ms=5000,
            config_snapshot_json="{}",
            final_answer="The answer is 42",
        )

        turn = await turn_repo.get(turn_id)
        assert turn is not None
        assert turn.final_answer == "The answer is 42"

        # upsert with final_answer=None should preserve existing value (COALESCE)
        await turn_repo.upsert(
            turn_id=turn_id,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            turn_index=0,
            user_input="query",
            status="completed",
            started_at_ms=1000,
            finished_at_ms=5000,
            config_snapshot_json="{}",
        )

        turn = await turn_repo.get(turn_id)
        assert turn is not None
        assert turn.final_answer == "The answer is 42"

    async def test_list_by_session_allows_missing_config_snapshot(
        self, turn_repo, session_repo
    ) -> None:
        """list_by_session(): preserves turns with no config snapshot."""
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
            user_input="query",
            status="running",
            started_at_ms=1000,
        )

        turns = await turn_repo.list_by_session(session_id)

        assert len(turns) == 1
        assert turns[0].id == turn_id
        assert turns[0].config_snapshot_json is None
