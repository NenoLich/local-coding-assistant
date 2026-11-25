"""Session repository unit tests."""

from __future__ import annotations

import uuid


class TestSessionRepo:
    async def test_create_inserts_session_with_all_columns(self, session_repo) -> None:
        """create(): inserts session, verifies all columns populated."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="test query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
            metadata_json='{"key": "value"}',
        )

        session = await session_repo.get(session_id)
        assert session is not None
        assert session.id == session_id
        assert session.user_input == "test query"
        assert session.status == "running"
        assert session.started_at_ms == 1000
        assert session.agent_mode == "frame_agent"
        assert session.model_name == "gpt-4"
        assert session.metadata_json == '{"key": "value"}'
        assert session.total_input_tokens == 0
        assert session.total_output_tokens == 0
        assert session.total_turns == 0
        assert session.total_frames == 0
        assert session.context_profile_json is None
        assert session.updated_at_ms is None

    async def test_get_returns_stored_session_with_all_fields(
        self, session_repo
    ) -> None:
        """get(): returns StoredSession with all fields including context_profile_json."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="idle",
            started_at_ms=1000,
            agent_mode="no_agent",
            model_name="gpt-4",
            metadata_json='{"meta": "data"}',
        )

        # Update context_profile_json via upsert (since create doesn't support it)
        await session_repo.upsert(
            session_id=session_id,
            status="idle",
            user_input="query",
            started_at_ms=1000,
            agent_mode="no_agent",
            model_name="gpt-4",
            metadata_json='{"meta": "data"}',
            context_profile_json='{"profile": "data"}',
            updated_at_ms=2000,
        )

        session = await session_repo.get(session_id)
        assert session is not None
        assert session.id == session_id
        assert session.context_profile_json == '{"profile": "data"}'
        assert session.updated_at_ms == 2000

    async def test_get_returns_none_for_missing_session(self, session_repo) -> None:
        session = await session_repo.get(str(uuid.uuid4()))
        assert session is None

    async def test_update_status_changes_status_and_updated_at(
        self, session_repo
    ) -> None:
        """update_status(): changes status, updates updated_at."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="idle",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        await session_repo.update_status(session_id, "running", updated_at_ms=2000)
        session = await session_repo.get(session_id)
        assert session is not None
        assert session.status == "running"
        assert session.updated_at_ms == 2000

    async def test_update_status_without_updated_at(self, session_repo) -> None:
        """update_status() without updated_at_ms leaves updated_at unchanged."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="idle",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        await session_repo.update_status(session_id, "running")
        session = await session_repo.get(session_id)
        assert session is not None
        assert session.status == "running"
        assert session.updated_at_ms is None

    async def test_update_totals_sets_absolute_values(self, session_repo) -> None:
        """update_totals(): sets absolute totals."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="idle",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        await session_repo.update_totals(
            session_id,
            total_input_tokens=60,
            total_output_tokens=40,
            total_turns=5,
            total_frames=10,
        )
        session = await session_repo.get(session_id)
        assert session is not None
        assert session.total_input_tokens == 60
        assert session.total_output_tokens == 40
        assert session.total_turns == 5
        assert session.total_frames == 10

    async def test_increment_totals_adds_deltas(self, session_repo) -> None:
        """increment_totals(): atomically adds deltas (tokens/turns/frames)."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="idle",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        # Set initial values
        await session_repo.update_totals(
            session_id,
            total_input_tokens=6,
            total_output_tokens=4,
            total_turns=2,
            total_frames=3,
        )

        # Increment
        await session_repo.increment_totals(
            session_id,
            total_input_tokens=3,
            total_output_tokens=2,
            total_turns=1,
            total_frames=2,
        )
        session = await session_repo.get(session_id)
        assert session is not None
        assert session.total_input_tokens == 9
        assert session.total_output_tokens == 6
        assert session.total_turns == 3
        assert session.total_frames == 5

    async def test_increment_totals_from_null(self, session_repo) -> None:
        """increment_totals() handles NULL initial values via COALESCE."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="idle",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        # Don't set initial values (they default to NULL in DB, but StoredSession returns 0)
        await session_repo.increment_totals(
            session_id,
            total_input_tokens=3,
            total_output_tokens=2,
            total_turns=1,
            total_frames=2,
        )
        session = await session_repo.get(session_id)
        assert session is not None
        assert session.total_input_tokens == 3
        assert session.total_output_tokens == 2
        assert session.total_turns == 1
        assert session.total_frames == 2

    async def test_list_recent_orders_by_started_at_desc(self, session_repo) -> None:
        """list_recent(): orders by started_at DESC, respects limit."""
        session1_id = str(uuid.uuid4())
        session2_id = str(uuid.uuid4())
        session3_id = str(uuid.uuid4())

        await session_repo.create(
            session_id=session1_id,
            user_input="query1",
            status="idle",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )
        await session_repo.create(
            session_id=session2_id,
            user_input="query2",
            status="idle",
            started_at_ms=3000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )
        await session_repo.create(
            session_id=session3_id,
            user_input="query3",
            status="idle",
            started_at_ms=2000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        sessions = await session_repo.list_recent(limit=10)
        assert len(sessions) == 3
        assert sessions[0].id == session2_id  # started_at=3000
        assert sessions[1].id == session3_id  # started_at=2000
        assert sessions[2].id == session1_id  # started_at=1000

    async def test_list_recent_respects_limit(self, session_repo) -> None:
        """list_recent(): respects limit parameter."""
        for i in range(5):
            await session_repo.create(
                session_id=str(uuid.uuid4()),
                user_input=f"query{i}",
                status="idle",
                started_at_ms=i * 1000,
                agent_mode="frame_agent",
                model_name="gpt-4",
            )

        sessions = await session_repo.list_recent(limit=3)
        assert len(sessions) == 3

    async def test_upsert_creates_new_session(self, session_repo) -> None:
        """upsert(): creates new session when ID doesn't exist."""
        session_id = str(uuid.uuid4())
        await session_repo.upsert(
            session_id=session_id,
            status="running",
            user_input="query",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        session = await session_repo.get(session_id)
        assert session is not None
        assert session.id == session_id
        assert session.status == "running"
        assert session.user_input == "query"
        assert session.started_at_ms == 1000

    async def test_upsert_status_only_update_preserves_fields(
        self, session_repo
    ) -> None:
        """upsert(): status-only update preserves user_input, agent_mode, started_at, metadata, context_profile."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="idle",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
            metadata_json='{"key": "value"}',
        )

        await session_repo.upsert(
            session_id=session_id,
            status="running",
            user_input="",  # sentinel for "no update"
            started_at_ms=0,  # sentinel for "no update"
            agent_mode="",  # sentinel for "no update"
        )

        session = await session_repo.get(session_id)
        assert session is not None
        assert session.status == "running"
        assert session.user_input == "query"  # preserved
        assert session.started_at_ms == 1000  # preserved
        assert session.agent_mode == "frame_agent"  # preserved
        assert session.model_name == "gpt-4"  # preserved
        assert session.metadata_json == '{"key": "value"}'  # preserved

    async def test_upsert_model_name_update_preserves_other_fields(
        self, session_repo
    ) -> None:
        """upsert(): model_name update preserves other fields."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="idle",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        await session_repo.upsert(
            session_id=session_id,
            status="idle",
            user_input="query",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4-turbo",
        )

        session = await session_repo.get(session_id)
        assert session is not None
        assert session.model_name == "gpt-4-turbo"
        assert session.user_input == "query"
        assert session.agent_mode == "frame_agent"

    async def test_upsert_total_null_is_noop(self, session_repo) -> None:
        """upsert(): total_* NULL = no-op (preserves existing values)."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="idle",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )
        await session_repo.update_totals(
            session_id,
            total_input_tokens=60,
            total_output_tokens=40,
            total_turns=5,
            total_frames=10,
        )

        await session_repo.upsert(
            session_id=session_id,
            status="running",
            user_input="query",
            started_at_ms=1000,
            agent_mode="frame_agent",
            total_input_tokens=None,  # NULL = no-op
            total_output_tokens=None,  # NULL = no-op
            total_turns=None,  # NULL = no-op
            total_frames=None,  # NULL = no-op
        )

        session = await session_repo.get(session_id)
        assert session is not None
        assert session.total_input_tokens == 60  # preserved
        assert session.total_output_tokens == 40  # preserved
        assert session.total_turns == 5  # preserved
        assert session.total_frames == 10  # preserved

    async def test_upsert_total_zero_sets_to_zero(self, session_repo) -> None:
        """upsert(): total_* 0 = set to 0 (overwrites existing)."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="idle",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )
        await session_repo.update_totals(
            session_id,
            total_input_tokens=60,
            total_output_tokens=40,
            total_turns=5,
            total_frames=10,
        )

        await session_repo.upsert(
            session_id=session_id,
            status="running",
            user_input="query",
            started_at_ms=1000,
            agent_mode="frame_agent",
            total_input_tokens=0,  # 0 = set to 0
            total_output_tokens=0,  # 0 = set to 0
            total_turns=0,  # 0 = set to 0
            total_frames=0,  # 0 = set to 0
        )

        session = await session_repo.get(session_id)
        assert session is not None
        assert session.total_input_tokens == 0  # overwritten
        assert session.total_output_tokens == 0  # overwritten
        assert session.total_turns == 0  # overwritten
        assert session.total_frames == 0  # overwritten

    async def test_delete_cascades_to_turns(self, session_repo, turn_repo) -> None:
        """delete(): cascades to turns (verify FK cascade)."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="idle",
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

        # Verify turn exists
        turn = await turn_repo.get(turn_id)
        assert turn is not None

        # Delete session
        await session_repo.delete(session_id)

        # Verify turn is deleted (cascade)
        turn = await turn_repo.get(turn_id)
        assert turn is None

    async def test_delete_removes_session(self, session_repo) -> None:
        """delete(): removes session."""
        session_id = str(uuid.uuid4())
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="idle",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        await session_repo.delete(session_id)
        session = await session_repo.get(session_id)
        assert session is None
