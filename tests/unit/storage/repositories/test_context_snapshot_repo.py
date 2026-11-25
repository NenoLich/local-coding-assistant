"""Context snapshot repository unit tests."""

from __future__ import annotations

import uuid

from local_coding_assistant.storage.contracts import StoredContextSnapshot


class TestContextSnapshotRepo:
    async def test_create_inserts_snapshot_with_all_fields(
        self, context_snapshot_repo, session_repo, turn_repo
    ) -> None:
        """create(): INSERT OR REPLACE, stores refs as JSON strings (or NULL)."""
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
        snapshot_id = str(uuid.uuid4())
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

        await context_snapshot_repo.create(
            snapshot_id=snapshot_id,
            turn_id=turn_id,
            schema_version=1,
            tool_set_ref='{"tool_ids": [1, 2]}',
            template_set_ref='{"template_ids": [3, 4]}',
            repo_snapshot_ref='{"hash": "abc123"}',
            memory_snapshot_ref=None,
            message_boundary_json='{"start_seq": 0, "end_seq": 10}',
            content_hash="hash123",
            payload_json='{"key": "value"}',
            created_at_ms=1000,
        )

        snapshot = await context_snapshot_repo.get_stored(snapshot_id)
        assert snapshot is not None
        assert snapshot.id == snapshot_id
        assert snapshot.turn_id == turn_id
        assert snapshot.schema_version == 1
        assert snapshot.tool_set_ref == '{"tool_ids": [1, 2]}'
        assert snapshot.template_set_ref == '{"template_ids": [3, 4]}'
        assert snapshot.repo_snapshot_ref == '{"hash": "abc123"}'
        assert snapshot.memory_snapshot_ref is None
        assert snapshot.message_boundary_json == '{"start_seq": 0, "end_seq": 10}'
        assert snapshot.content_hash == "hash123"
        assert snapshot.payload_json == '{"key": "value"}'
        assert snapshot.created_at_ms == 1000

    async def test_get_returns_context_snapshot_parsed_via_to_runtime(
        self, context_snapshot_repo, session_repo, turn_repo
    ) -> None:
        """get(): returns ContextSnapshot (parsed refs via to_runtime())."""
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
        snapshot_id = str(uuid.uuid4())
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

        await context_snapshot_repo.create(
            snapshot_id=snapshot_id,
            turn_id=turn_id,
            tool_set_ref='{"tool_ids": [1, 2]}',
            content_hash="hash123",
        )

        snapshot = await context_snapshot_repo.get(snapshot_id)
        assert snapshot is not None
        assert snapshot.id == snapshot_id
        assert snapshot.turn_id == turn_id
        assert snapshot.content_hash == "hash123"
        # to_runtime() should have parsed the JSON refs
        assert snapshot.tool_set_ref is not None

    async def test_get_returns_none_for_missing_snapshot(
        self, context_snapshot_repo
    ) -> None:
        snapshot = await context_snapshot_repo.get(str(uuid.uuid4()))
        assert snapshot is None

    async def test_get_by_turn_returns_context_snapshot(
        self, context_snapshot_repo, session_repo, turn_repo
    ) -> None:
        """get_by_turn(): returns ContextSnapshot for turn."""
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
        snapshot_id = str(uuid.uuid4())
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

        await context_snapshot_repo.create(
            snapshot_id=snapshot_id,
            turn_id=turn_id,
            content_hash="hash123",
        )

        snapshot = await context_snapshot_repo.get_by_turn(turn_id)
        assert snapshot is not None
        assert snapshot.id == snapshot_id
        assert snapshot.turn_id == turn_id

    async def test_get_by_turn_returns_none_for_missing_turn(
        self, context_snapshot_repo
    ) -> None:
        snapshot = await context_snapshot_repo.get_by_turn(str(uuid.uuid4()))
        assert snapshot is None

    async def test_get_stored_returns_raw_json_refs(
        self, context_snapshot_repo, session_repo, turn_repo
    ) -> None:
        """get_stored(): returns StoredContextSnapshot (raw JSON refs)."""
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
        snapshot_id = str(uuid.uuid4())
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

        await context_snapshot_repo.create(
            snapshot_id=snapshot_id,
            turn_id=turn_id,
            tool_set_ref='{"tool_ids": [1, 2]}',
            content_hash="hash123",
        )

        snapshot = await context_snapshot_repo.get_stored(snapshot_id)
        assert snapshot is not None
        assert isinstance(snapshot, StoredContextSnapshot)
        assert snapshot.tool_set_ref == '{"tool_ids": [1, 2]}'

    async def test_get_stored_by_turn_returns_stored_context_snapshot(
        self, context_snapshot_repo, session_repo, turn_repo
    ) -> None:
        """get_stored_by_turn(): returns StoredContextSnapshot by turn ID."""
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
        snapshot_id = str(uuid.uuid4())
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

        await context_snapshot_repo.create(
            snapshot_id=snapshot_id,
            turn_id=turn_id,
            tool_set_ref='{"tool_ids": [1, 2]}',
            content_hash="hash123",
        )

        snapshot = await context_snapshot_repo.get_stored_by_turn(turn_id)
        assert snapshot is not None
        assert isinstance(snapshot, StoredContextSnapshot)
        assert snapshot.id == snapshot_id
        assert snapshot.tool_set_ref == '{"tool_ids": [1, 2]}'

    async def test_upsert_creates_new_snapshot(
        self, context_snapshot_repo, session_repo, turn_repo
    ) -> None:
        """upsert(): creates new snapshot when ID doesn't exist."""
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
        snapshot_id = str(uuid.uuid4())
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

        await context_snapshot_repo.upsert(
            snapshot_id=snapshot_id,
            turn_id=turn_id,
            tool_set_ref='{"tool_ids": [1]}',
            content_hash="hash123",
        )

        snapshot = await context_snapshot_repo.get_stored(snapshot_id)
        assert snapshot is not None
        assert snapshot.tool_set_ref == '{"tool_ids": [1]}'

    async def test_upsert_updates_existing_snapshot(
        self, context_snapshot_repo, session_repo, turn_repo
    ) -> None:
        """upsert(): INSERT ON CONFLICT DO UPDATE, merges all ref fields."""
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
        snapshot_id = str(uuid.uuid4())
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

        await context_snapshot_repo.create(
            snapshot_id=snapshot_id,
            turn_id=turn_id,
            tool_set_ref='{"tool_ids": [1]}',
            template_set_ref='{"template_ids": [3]}',
            content_hash="hash123",
        )

        await context_snapshot_repo.upsert(
            snapshot_id=snapshot_id,
            turn_id=turn_id,
            tool_set_ref='{"tool_ids": [1, 2]}',
            template_set_ref='{"template_ids": [3, 4]}',
            repo_snapshot_ref='{"hash": "abc"}',
            content_hash="hash456",
        )

        snapshot = await context_snapshot_repo.get_stored(snapshot_id)
        assert snapshot is not None
        assert snapshot.tool_set_ref == '{"tool_ids": [1, 2]}'
        assert snapshot.template_set_ref == '{"template_ids": [3, 4]}'
        assert snapshot.repo_snapshot_ref == '{"hash": "abc"}'
        assert snapshot.content_hash == "hash456"

    async def test_to_runtime_normalizes_null_string_to_none(
        self, context_snapshot_repo, session_repo, turn_repo
    ) -> None:
        """to_runtime(): handles "null"/""/NULL ref normalization correctly."""
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
        snapshot_id = str(uuid.uuid4())
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

        # Store literal "null" string (old format)
        await context_snapshot_repo.create(
            snapshot_id=snapshot_id,
            turn_id=turn_id,
            tool_set_ref="null",
            template_set_ref='""',
            repo_snapshot_ref="",
            content_hash="hash123",
        )

        snapshot = await context_snapshot_repo.get(snapshot_id)
        assert snapshot is not None
        # to_runtime() should normalize "null", '""', "" to None
        # The refs should be empty/default, not parsed from the literal strings
        assert snapshot.tool_set_ref is not None  # Should be empty ToolSetRef
        assert snapshot.template_set_ref is not None  # Should be empty TemplateSetRef
        assert snapshot.repo_snapshot_ref is None  # Should be None

    async def test_create_auto_fills_created_at(
        self, context_snapshot_repo, session_repo, turn_repo
    ) -> None:
        """create(): auto-fills created_at_ms when None."""
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
        snapshot_id = str(uuid.uuid4())
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

        from datetime import UTC, datetime

        before = int(datetime.now(UTC).timestamp() * 1000)
        await context_snapshot_repo.create(
            snapshot_id=snapshot_id,
            turn_id=turn_id,
            content_hash="hash123",
            created_at_ms=None,
        )
        after = int(datetime.now(UTC).timestamp() * 1000)

        snapshot = await context_snapshot_repo.get_stored(snapshot_id)
        assert snapshot is not None
        assert snapshot.created_at_ms is not None
        assert before <= snapshot.created_at_ms <= after

    async def test_upsert_auto_fills_created_at(
        self, context_snapshot_repo, session_repo, turn_repo
    ) -> None:
        """upsert(): auto-fills created_at_ms when None."""
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
        snapshot_id = str(uuid.uuid4())
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

        from datetime import UTC, datetime

        before = int(datetime.now(UTC).timestamp() * 1000)
        await context_snapshot_repo.upsert(
            snapshot_id=snapshot_id,
            turn_id=turn_id,
            content_hash="hash123",
            created_at_ms=None,
        )
        after = int(datetime.now(UTC).timestamp() * 1000)

        snapshot = await context_snapshot_repo.get_stored(snapshot_id)
        assert snapshot is not None
        assert snapshot.created_at_ms is not None
        assert before <= snapshot.created_at_ms <= after
