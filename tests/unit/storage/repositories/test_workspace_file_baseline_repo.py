"""Workspace file baseline repository unit tests."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime


class TestWorkspaceFileBaselineRepo:
    async def test_upsert_inserts_new_baseline(
        self, workspace_file_baseline_repo, session_repo
    ) -> None:
        """upsert(): INSERT ON CONFLICT(workspace_id, session_id, file_path) DO UPDATE."""
        workspace_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())

        # Create session first (foreign key constraint)
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        observation = {
            "path": "file.txt",
            "hash": "abc123",
            "size": 100,
            "mtime": datetime.now(UTC).isoformat(),
            "inode": "inode123",
            "source": {"value": "read"},
            "observed_at": datetime.now(UTC).isoformat(),
        }

        await workspace_file_baseline_repo.upsert(
            workspace_id=workspace_id,
            observation=observation,
            turn_id=None,
            frame_id=None,
            session_id=session_id,
        )
        await workspace_file_baseline_repo._conn.commit()

        baselines = await workspace_file_baseline_repo.load(workspace_id, session_id)
        assert len(baselines) == 1
        assert baselines[0]["path"] == "file.txt"
        assert baselines[0]["hash"] == "abc123"

    async def test_upsert_updates_existing_baseline(
        self, workspace_file_baseline_repo, session_repo
    ) -> None:
        """upsert(): updates hash, size, mtime, inode, source, observed_at, turn_id, frame_id."""
        workspace_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        turn_id = str(uuid.uuid4())
        frame_id = str(uuid.uuid4())

        # Create session first (foreign key constraint)
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        observation_1 = {
            "path": "file.txt",
            "hash": "abc123",
            "size": 100,
            "mtime": datetime.now(UTC).isoformat(),
            "inode": "inode123",
            "source": {"value": "read"},
            "observed_at": datetime.now(UTC).isoformat(),
        }

        await workspace_file_baseline_repo.upsert(
            workspace_id=workspace_id,
            observation=observation_1,
            turn_id=None,
            frame_id=None,
            session_id=session_id,
        )
        await workspace_file_baseline_repo._conn.commit()

        # Update with new values
        observation_2 = {
            "path": "file.txt",
            "hash": "def456",
            "size": 200,
            "mtime": datetime.now(UTC).isoformat(),
            "inode": "inode456",
            "source": {"value": "write"},
            "observed_at": datetime.now(UTC).isoformat(),
        }

        await workspace_file_baseline_repo.upsert(
            workspace_id=workspace_id,
            observation=observation_2,
            turn_id=turn_id,
            frame_id=frame_id,
            session_id=session_id,
        )
        await workspace_file_baseline_repo._conn.commit()

        baselines = await workspace_file_baseline_repo.load(workspace_id, session_id)
        # WRITE source is filtered out by load; verify via raw SQL instead
        cursor = await workspace_file_baseline_repo._conn.execute(
            "SELECT hash, size FROM workspace_file_baselines WHERE workspace_id = ? AND session_id = ?",
            (workspace_id, session_id),
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == "def456"
        assert row[1] == 200

    async def test_load_filters_by_workspace_id(
        self, workspace_file_baseline_repo, session_repo
    ) -> None:
        """load(): filters by workspace_id (+ optional session_id)."""
        workspace_id_1 = str(uuid.uuid4())
        workspace_id_2 = str(uuid.uuid4())
        session_id = str(uuid.uuid4())

        # Create session first (foreign key constraint)
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        observation = {
            "path": "file.txt",
            "hash": "abc123",
            "size": 100,
            "mtime": datetime.now(UTC).isoformat(),
            "source": {"value": "read"},
            "observed_at": datetime.now(UTC).isoformat(),
        }

        await workspace_file_baseline_repo.upsert(
            workspace_id=workspace_id_1,
            observation=observation,
            turn_id=None,
            frame_id=None,
            session_id=session_id,
        )
        await workspace_file_baseline_repo._conn.commit()

        await workspace_file_baseline_repo.upsert(
            workspace_id=workspace_id_2,
            observation=observation,
            turn_id=None,
            frame_id=None,
            session_id=session_id,
        )
        await workspace_file_baseline_repo._conn.commit()

        baselines_1 = await workspace_file_baseline_repo.load(workspace_id_1)
        baselines_2 = await workspace_file_baseline_repo.load(workspace_id_2)

        assert len(baselines_1) == 1
        assert len(baselines_2) == 1

    async def test_load_filters_by_session_id(
        self, workspace_file_baseline_repo, session_repo
    ) -> None:
        """load(): filters by session_id when provided."""
        workspace_id = str(uuid.uuid4())
        session_id_1 = str(uuid.uuid4())
        session_id_2 = str(uuid.uuid4())

        # Create sessions first (foreign key constraint)
        await session_repo.create(
            session_id=session_id_1,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )
        await session_repo.create(
            session_id=session_id_2,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        observation = {
            "path": "file.txt",
            "hash": "abc123",
            "size": 100,
            "mtime": datetime.now(UTC).isoformat(),
            "source": {"value": "read"},
            "observed_at": datetime.now(UTC).isoformat(),
        }

        await workspace_file_baseline_repo.upsert(
            workspace_id=workspace_id,
            observation=observation,
            turn_id=None,
            frame_id=None,
            session_id=session_id_1,
        )
        await workspace_file_baseline_repo._conn.commit()

        await workspace_file_baseline_repo.upsert(
            workspace_id=workspace_id,
            observation=observation,
            turn_id=None,
            frame_id=None,
            session_id=session_id_2,
        )
        await workspace_file_baseline_repo._conn.commit()

        baselines_1 = await workspace_file_baseline_repo.load(
            workspace_id, session_id_1
        )
        baselines_2 = await workspace_file_baseline_repo.load(
            workspace_id, session_id_2
        )
        baselines_all = await workspace_file_baseline_repo.load(workspace_id)

        assert len(baselines_1) == 1
        assert len(baselines_2) == 1
        assert len(baselines_all) == 2

    async def test_load_reconstructs_file_observation(
        self, workspace_file_baseline_repo, session_repo
    ) -> None:
        """load(): reconstructs dict observation."""
        workspace_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())

        # Create session first (foreign key constraint)
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        observation = {
            "path": "file.txt",
            "hash": "abc123",
            "size": 100,
            "mtime": datetime.now(UTC).isoformat(),
            "inode": "inode123",
            "source": {"value": "snapshot"},
            "observed_at": datetime.now(UTC).isoformat(),
        }

        await workspace_file_baseline_repo.upsert(
            workspace_id=workspace_id,
            observation=observation,
            turn_id=None,
            frame_id=None,
            session_id=session_id,
        )
        await workspace_file_baseline_repo._conn.commit()

        baselines = await workspace_file_baseline_repo.load(workspace_id, session_id)
        assert len(baselines) == 1
        assert isinstance(baselines[0], dict)
        assert baselines[0]["path"] == "file.txt"
        assert baselines[0]["hash"] == "abc123"
        assert baselines[0]["size"] == 100
        assert baselines[0]["inode"] == "inode123"
        assert baselines[0]["source"]["value"] == "snapshot"

    async def test_get_by_turn_id_returns_raw_tuples(
        self, workspace_file_baseline_repo, session_repo
    ) -> None:
        """get_by_turn_id(): returns raw tuples for specific turn_ids in session."""
        workspace_id_1 = str(uuid.uuid4())
        workspace_id_2 = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        turn_id_1 = "turn-123"
        turn_id_2 = "turn-456"
        frame_id = str(uuid.uuid4())

        # Create session first (foreign key constraint)
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        observation = {
            "path": "file.txt",
            "hash": "abc123",
            "size": 100,
            "mtime": datetime.now(UTC).isoformat(),
            "source": {"value": "read"},
            "observed_at": datetime.now(UTC).isoformat(),
        }

        await workspace_file_baseline_repo.upsert(
            workspace_id=workspace_id_1,
            observation=observation,
            turn_id=turn_id_1,
            frame_id=frame_id,
            session_id=session_id,
        )
        await workspace_file_baseline_repo._conn.commit()

        await workspace_file_baseline_repo.upsert(
            workspace_id=workspace_id_2,
            observation=observation,
            turn_id=turn_id_2,
            frame_id=frame_id,
            session_id=session_id,
        )
        await workspace_file_baseline_repo._conn.commit()

        rows = await workspace_file_baseline_repo.get_by_turn_id(
            session_id, {turn_id_1}
        )
        assert len(rows) == 1
        assert rows[0][8] == turn_id_1  # turn_id is at index 8

    async def test_get_by_turn_id_empty_set(self, workspace_file_baseline_repo) -> None:
        """get_by_turn_id(): returns empty list for empty turn_ids."""
        session_id = str(uuid.uuid4())
        rows = await workspace_file_baseline_repo.get_by_turn_id(session_id, set())
        assert rows == []

    async def test_delete_removes_baseline(
        self, workspace_file_baseline_repo, session_repo
    ) -> None:
        """delete(): removes by workspace_id + file_path."""
        workspace_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())

        # Create session first (foreign key constraint)
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        observation = {
            "path": "file.txt",
            "hash": "abc123",
            "size": 100,
            "mtime": datetime.now(UTC).isoformat(),
            "source": {"value": "read"},
            "observed_at": datetime.now(UTC).isoformat(),
        }

        await workspace_file_baseline_repo.upsert(
            workspace_id=workspace_id,
            observation=observation,
            turn_id=None,
            frame_id=None,
            session_id=session_id,
        )
        await workspace_file_baseline_repo._conn.commit()

        baselines = await workspace_file_baseline_repo.load(workspace_id)
        assert len(baselines) == 1

        await workspace_file_baseline_repo.delete(workspace_id, "file.txt")

        baselines = await workspace_file_baseline_repo.load(workspace_id)
        assert len(baselines) == 0

    async def test_upsert_handles_none_turn_and_frame_id(
        self, workspace_file_baseline_repo, session_repo
    ) -> None:
        """upsert(): handles None turn_id and frame_id."""
        workspace_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())

        # Create session first (foreign key constraint)
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        observation = {
            "path": "file.txt",
            "hash": "abc123",
            "size": 100,
            "mtime": datetime.now(UTC).isoformat(),
            "source": {"value": "read"},
            "observed_at": datetime.now(UTC).isoformat(),
        }

        await workspace_file_baseline_repo.upsert(
            workspace_id=workspace_id,
            observation=observation,
            turn_id=None,
            frame_id=None,
            session_id=session_id,
        )
        await workspace_file_baseline_repo._conn.commit()

        baselines = await workspace_file_baseline_repo.load(workspace_id, session_id)
        assert len(baselines) == 1

    async def test_upsert_handles_none_turn_and_frame_id(
        self, workspace_file_baseline_repo, session_repo
    ) -> None:
        """upsert(): handles None turn_id and frame_id."""
        workspace_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())

        # Create session first (foreign key constraint)
        await session_repo.create(
            session_id=session_id,
            user_input="query",
            status="running",
            started_at_ms=1000,
            agent_mode="frame_agent",
            model_name="gpt-4",
        )

        observation = {
            "path": "file.txt",
            "hash": "abc123",
            "size": 100,
            "mtime": datetime.now(UTC).isoformat(),
            "source": {"value": "read"},
            "observed_at": datetime.now(UTC).isoformat(),
        }

        await workspace_file_baseline_repo.upsert(
            workspace_id=workspace_id,
            observation=observation,
            turn_id=None,
            frame_id=None,
            session_id=session_id,
        )
        await workspace_file_baseline_repo._conn.commit()

        baselines = await workspace_file_baseline_repo.load(workspace_id, session_id)
        assert len(baselines) == 1

    async def test_load_returns_empty_list_for_no_baselines(
        self, workspace_file_baseline_repo
    ) -> None:
        """load(): returns empty list when no baselines exist."""
        workspace_id = str(uuid.uuid4())
        baselines = await workspace_file_baseline_repo.load(workspace_id)
        assert baselines == []
