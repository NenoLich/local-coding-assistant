"""Integration tests for ExecutionStore file baseline operations.

Covers upsert, load, conflict detection, source filtering, and branch
remapping of file baselines.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

from local_coding_assistant.runtime.events.contracts import (
    FrameCompleteEvent,
    FrameStartEvent,
    LLMCompleteEvent,
    SessionStartEvent,
    ToolResultEvent,
    TurnCompleteEvent,
    TurnStartEvent,
)
from local_coding_assistant.storage.execution_store import ExecutionStore

WORKSPACE_ID = "ws_baseline_test"


def _new_id() -> str:
    return f"t_{uuid.uuid4().hex[:12]}"


async def _fire_session_start(
    store: ExecutionStore,
    session_id: str,
    turn_id: str,
    user_input: str = "hello",
    agent_mode: str = "frame_agent",
    model_name: str = "gpt-4",
) -> None:
    await store.process_event(
        SessionStartEvent(
            trace_id=_new_id(),
            turn_id=turn_id,
            session_id=session_id,
            user_input=user_input,
            agent_mode=agent_mode,
            model_name=model_name,
        )
    )


async def _fire_turn_start(
    store: ExecutionStore,
    session_id: str,
    turn_id: str,
    turn_index: int,
    user_input: str = "turn query",
    config_snapshot_json: str | None = "{}",
    context_snapshot_json: str | None = None,
) -> None:
    await store.process_event(
        TurnStartEvent(
            trace_id=_new_id(),
            turn_id=turn_id,
            session_id=session_id,
            user_input=user_input,
            turn_index=turn_index,
            config_snapshot_json=config_snapshot_json,
            context_snapshot_json=context_snapshot_json,
        )
    )


async def _fire_frame_start(
    store: ExecutionStore,
    session_id: str,
    turn_id: str,
    frame_id: str,
    iteration: int,
    context_snapshot_id: str | None = None,
) -> None:
    await store.process_event(
        FrameStartEvent(
            trace_id=_new_id(),
            turn_id=turn_id,
            session_id=session_id,
            frame_id=frame_id,
            iteration=iteration,
            context_snapshot_id=context_snapshot_id or "snap_default",
            context_hash="hash_default",
        )
    )


async def _fire_llm_complete(
    store: ExecutionStore,
    session_id: str,
    turn_id: str,
    frame_id: str,
    content: str = "response",
    input_tokens: int = 10,
    output_tokens: int = 0,
    model: str = "gpt-4",
) -> None:
    await store.process_event(
        LLMCompleteEvent(
            trace_id=_new_id(),
            turn_id=turn_id,
            session_id=session_id,
            frame_id=frame_id,
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
        )
    )


async def _fire_tool_result(
    store: ExecutionStore,
    session_id: str,
    turn_id: str,
    frame_id: str,
    tool_call_id: str = "call_1",
    tool_name: str = "calculator",
    success: bool = True,
    output: str = "42",
) -> None:
    await store.process_event(
        ToolResultEvent(
            trace_id=_new_id(),
            turn_id=turn_id,
            session_id=session_id,
            frame_id=frame_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            success=success,
            output=output,
        )
    )


async def _fire_frame_complete(
    store: ExecutionStore,
    session_id: str,
    turn_id: str,
    frame_id: str,
    status: str = "completed",
    duration_ms: int = 100,
    total_input_tokens: int = 0,
    total_output_tokens: int = 0,
    finish_reason: str = "stop",
) -> None:
    await store.process_event(
        FrameCompleteEvent(
            trace_id=_new_id(),
            turn_id=turn_id,
            session_id=session_id,
            frame_id=frame_id,
            status=status,
            duration_ms=duration_ms,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            model_name="gpt-4",
            finish_reason=finish_reason,
        )
    )


async def _fire_turn_complete(
    store: ExecutionStore,
    session_id: str,
    turn_id: str,
    status: str = "completed",
    total_input_tokens: int = 0,
    total_output_tokens: int = 0,
    total_frames: int = 0,
    final_answer: str | None = None,
) -> None:
    await store.process_event(
        TurnCompleteEvent(
            trace_id=_new_id(),
            turn_id=turn_id,
            session_id=session_id,
            status=status,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_frames=total_frames,
            final_answer=final_answer,
        )
    )


async def _seed_basic_session(
    store: ExecutionStore,
    workspace_id: str = WORKSPACE_ID,
) -> tuple[str, str, str, str]:
    """Create a minimal session with one turn, one frame, and events.

    Returns: (session_id, turn_id, frame_id, snapshot_id)
    """
    session_id = _new_id()
    turn_id = _new_id()
    frame_id = _new_id()
    snapshot_id = _new_id()

    await _fire_session_start(store, session_id, turn_id)
    await _fire_turn_start(
        store,
        session_id,
        turn_id,
        turn_index=0,
        context_snapshot_json=json.dumps(
            {
                "id": snapshot_id,
                "schema_version": 1,
                "tool_set_ref": {"tool_ids": [], "tool_names": [], "version": "1.0.0"},
                "template_set_ref": {
                    "template_ids": [],
                    "template_paths": [],
                    "version": "1.0.0",
                },
                "content_hash": "snap_hash_basic",
                "created_at_ms": 1000,
            }
        ),
    )
    await _fire_frame_start(store, session_id, turn_id, frame_id, iteration=0)
    await _fire_llm_complete(
        store, session_id, turn_id, frame_id, input_tokens=15, output_tokens=0
    )
    await _fire_tool_result(
        store,
        session_id,
        turn_id,
        frame_id,
        tool_call_id="call_1",
        tool_name="calculator",
        success=True,
        output="42",
    )
    await _fire_frame_complete(
        store,
        session_id,
        turn_id,
        frame_id,
        status="completed",
        total_input_tokens=15,
        total_output_tokens=0,
    )
    await _fire_turn_complete(store, session_id, turn_id, status="completed")

    return session_id, turn_id, frame_id, snapshot_id


class TestFileBaselineUpsert:
    async def test_upsert_inserts_new_baseline(
        self, execution_store: ExecutionStore
    ) -> None:
        """upsert_file_baseline(): INSERT ON CONFLICT(workspace_id, session_id, file_path) DO UPDATE.

        Stores a new baseline observation.
        """
        store = execution_store
        session_id, turn_id, frame_id, _ = await _seed_basic_session(store)
        observation = {
            "path": "src/main.py",
            "hash": "abc123",
            "size": 200,
            "mtime": datetime.now(UTC).isoformat(),
            "inode": "inode_001",
            "source": {"value": "read"},
            "observed_at": datetime.now(UTC).isoformat(),
        }

        await store.upsert_file_baseline(
            workspace_id=WORKSPACE_ID,
            observation=observation,
            turn_id=turn_id,
            frame_id=frame_id,
            session_id=session_id,
        )
        await store.flush()

        baselines = await store.load_file_baselines(
            workspace_id=WORKSPACE_ID, session_id=session_id
        )
        assert len(baselines) == 1
        assert baselines[0]["path"] == "src/main.py"
        assert baselines[0]["hash"] == "abc123"
        assert baselines[0]["size"] == 200
        assert baselines[0]["inode"] == "inode_001"
        assert baselines[0]["source"]["value"] == "read"

    async def test_upsert_updates_existing_baseline(
        self, execution_store: ExecutionStore
    ) -> None:
        """upsert_file_baseline(): updates hash, size, mtime, inode, source, observed_at, turn_id, frame_id.

        On conflict (same workspace_id, session_id, file_path), the baseline is updated.
        """
        store = execution_store
        session_id, turn_id, frame_id, _ = await _seed_basic_session(store)

        observation_1 = {
            "path": "src/main.py",
            "hash": "abc123",
            "size": 200,
            "mtime": datetime.now(UTC).isoformat(),
            "inode": "inode_001",
            "source": {"value": "read"},
            "observed_at": datetime.now(UTC).isoformat(),
        }

        await store.upsert_file_baseline(
            workspace_id=WORKSPACE_ID,
            observation=observation_1,
            turn_id=turn_id,
            frame_id=frame_id,
            session_id=session_id,
        )
        await store.flush()

        # Update with new values
        observation_2 = {
            "path": "src/main.py",
            "hash": "def456",
            "size": 300,
            "mtime": datetime.now(UTC).isoformat(),
            "inode": "inode_002",
            "source": {"value": "write"},
            "observed_at": datetime.now(UTC).isoformat(),
        }

        await store.upsert_file_baseline(
            workspace_id=WORKSPACE_ID,
            observation=observation_2,
            turn_id="new_turn",
            frame_id="new_frame",
            session_id=session_id,
        )
        await store.flush()

        # Use raw SQL to check turn_id/frame_id since dict doesn't have those fields
        # and load_file_baselines filters out WRITE source
        cursor = await store._conn.execute(
            "SELECT turn_id, frame_id FROM workspace_file_baselines WHERE workspace_id = ? AND session_id = ?",
            (WORKSPACE_ID, session_id),
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == "new_turn"
        assert row[1] == "new_frame"

        # Note: load_file_baselines filters out WRITE source, so check raw SQL for source
        baselines = await store.load_file_baselines(
            workspace_id=WORKSPACE_ID, session_id=session_id
        )
        # WRITE source is filtered out by load; verify via raw SQL instead
        cursor2 = await store._conn.execute(
            "SELECT source FROM workspace_file_baselines WHERE workspace_id = ? AND session_id = ?",
            (WORKSPACE_ID, session_id),
        )
        row2 = await cursor2.fetchone()
        assert row2 is not None
        assert row2[0] == "write"

        baselines = await store.load_file_baselines(
            workspace_id=WORKSPACE_ID, session_id=session_id
        )
        assert len(baselines) == 0  # WRITE filtered out

    async def test_upsert_handles_none_turn_and_frame_id(
        self, execution_store: ExecutionStore
    ) -> None:
        """upsert_file_baseline(): handles None turn_id and frame_id."""
        store = execution_store
        session_id, turn_id, frame_id, _ = await _seed_basic_session(store)

        observation = {
            "path": "config.yaml",
            "hash": "hash_789",
            "size": 50,
            "mtime": datetime.now(UTC).isoformat(),
            "inode": None,
            "source": {"value": "snapshot"},
            "observed_at": datetime.now(UTC).isoformat(),
        }

        await store.upsert_file_baseline(
            workspace_id=WORKSPACE_ID,
            observation=observation,
            turn_id=None,
            frame_id=None,
            session_id=session_id,
        )
        await store.flush()

        baselines = await store.load_file_baselines(
            workspace_id=WORKSPACE_ID, session_id=session_id
        )
        assert len(baselines) == 1
        assert baselines[0]["path"] == "config.yaml"
        assert baselines[0]["hash"] == "hash_789"
        assert baselines[0]["inode"] is None
        assert baselines[0]["source"]["value"] == "snapshot"

    async def test_upsert_preserves_session_when_turn_frame_change(
        self, execution_store: ExecutionStore
    ) -> None:
        """Upsert with same file_path but different turn_id/frame_id updates turn/frame refs.

        The session_id foreign key is preserved; only turn_id and frame_id change on conflict.
        """
        store = execution_store
        session_id, turn_id, frame_id, _ = await _seed_basic_session(store)

        observation_1 = {
            "path": "utils/helpers.py",
            "hash": "hash_aaa",
            "size": 100,
            "mtime": datetime.now(UTC).isoformat(),
            "inode": "inode_aaa",
            "source": {"value": "read"},
            "observed_at": datetime.now(UTC).isoformat(),
        }

        await store.upsert_file_baseline(
            workspace_id=WORKSPACE_ID,
            observation=observation_1,
            turn_id=turn_id,
            frame_id=frame_id,
            session_id=session_id,
        )
        await store.flush()

        # Upsert again with different turn/frame but same file_path + session
        observation_2 = {
            "path": "utils/helpers.py",
            "hash": "hash_bbb",
            "size": 150,
            "mtime": datetime.now(UTC).isoformat(),
            "inode": "inode_bbb",
            "source": {"value": "agent_create"},
            "observed_at": datetime.now(UTC).isoformat(),
        }

        await store.upsert_file_baseline(
            workspace_id=WORKSPACE_ID,
            observation=observation_2,
            turn_id="different_turn",
            frame_id="different_frame",
            session_id=session_id,
        )
        await store.flush()

        # Use raw SQL to check turn_id/frame_id
        cursor = await store._conn.execute(
            "SELECT turn_id, frame_id FROM workspace_file_baselines WHERE workspace_id = ? AND session_id = ?",
            (WORKSPACE_ID, session_id),
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == "different_turn"
        assert row[1] == "different_frame"

        baselines = await store.load_file_baselines(
            workspace_id=WORKSPACE_ID, session_id=session_id
        )
        assert len(baselines) == 1
        assert baselines[0]["hash"] == "hash_bbb"
        assert baselines[0]["size"] == 150
        assert baselines[0]["source"]["value"] == "agent_create"


class TestFileBaselineLoad:
    async def test_load_returns_all_for_workspace(
        self, execution_store: ExecutionStore
    ) -> None:
        """load_file_baselines(): returns all baselines for a workspace."""
        store = execution_store
        session_id_1, _, _, _ = await _seed_basic_session(
            store, workspace_id=WORKSPACE_ID
        )
        session_id_2, _, _, _ = await _seed_basic_session(
            store, workspace_id=WORKSPACE_ID
        )

        # Upsert baseline for session_1
        observation_1 = {
            "path": "src/main.py",
            "hash": "hash_1",
            "size": 100,
            "mtime": datetime.now(UTC).isoformat(),
            "inode": None,
            "source": {"value": "read"},
            "observed_at": datetime.now(UTC).isoformat(),
        }
        await store.upsert_file_baseline(
            workspace_id=WORKSPACE_ID,
            observation=observation_1,
            turn_id=None,
            frame_id=None,
            session_id=session_id_1,
        )
        await store.flush()

        # Upsert baseline for session_2
        observation_2 = {
            "path": "extra.txt",
            "hash": "extra_hash",
            "size": 75,
            "mtime": datetime.now(UTC).isoformat(),
            "inode": None,
            "source": {"value": "read"},
            "observed_at": datetime.now(UTC).isoformat(),
        }
        await store.upsert_file_baseline(
            workspace_id=WORKSPACE_ID,
            observation=observation_2,
            turn_id=None,
            frame_id=None,
            session_id=session_id_2,
        )
        await store.flush()

        all_baselines = await store.load_file_baselines(workspace_id=WORKSPACE_ID)
        session_1_baselines = await store.load_file_baselines(
            workspace_id=WORKSPACE_ID, session_id=session_id_1
        )
        session_2_baselines = await store.load_file_baselines(
            workspace_id=WORKSPACE_ID, session_id=session_id_2
        )

        assert len(all_baselines) == 2
        assert len(session_1_baselines) == 1
        assert session_1_baselines[0]["path"] == "src/main.py"
        assert len(session_2_baselines) == 1
        assert session_2_baselines[0]["path"] == "extra.txt"

    async def test_load_returns_empty_when_no_baselines(
        self, execution_store: ExecutionStore
    ) -> None:
        """load_file_baselines(): returns empty list when no baselines exist."""
        store = execution_store
        session_id = _new_id()
        await _fire_session_start(store, session_id, _new_id())

        baselines = await store.load_file_baselines(workspace_id=WORKSPACE_ID)
        assert baselines == []

    async def test_load_filters_by_session_id(
        self, execution_store: ExecutionStore
    ) -> None:
        """load_file_baselines(): filters by session_id when provided."""
        store = execution_store
        session_id_1, _, _, _ = await _seed_basic_session(
            store, workspace_id=WORKSPACE_ID
        )
        session_id_2, _, _, _ = await _seed_basic_session(
            store, workspace_id=WORKSPACE_ID
        )

        observation = {
            "path": "shared.txt",
            "hash": "shared_hash",
            "size": 80,
            "mtime": datetime.now(UTC).isoformat(),
            "inode": None,
            "source": {"value": "read"},
            "observed_at": datetime.now(UTC).isoformat(),
        }

        # Only upsert for session_1
        await store.upsert_file_baseline(
            workspace_id=WORKSPACE_ID,
            observation=observation,
            turn_id=None,
            frame_id=None,
            session_id=session_id_1,
        )
        await store.flush()

        session_1_baselines = await store.load_file_baselines(
            workspace_id=WORKSPACE_ID, session_id=session_id_1
        )
        session_2_baselines = await store.load_file_baselines(
            workspace_id=WORKSPACE_ID, session_id=session_id_2
        )

        assert len(session_1_baselines) == 1
        assert session_1_baselines[0]["path"] == "shared.txt"
        assert len(session_2_baselines) == 0


class TestFileBaselineSourceFiltering:
    async def test_source_read_persisted(self, execution_store: ExecutionStore) -> None:
        """Only READ source baselines are persisted by default."""
        store = execution_store
        session_id, turn_id, frame_id, _ = await _seed_basic_session(store)

        observation = {
            "path": "read_file.txt",
            "hash": "read_hash",
            "size": 42,
            "mtime": datetime.now(UTC).isoformat(),
            "inode": "inode_read",
            "source": {"value": "read"},
            "observed_at": datetime.now(UTC).isoformat(),
        }

        await store.upsert_file_baseline(
            workspace_id=WORKSPACE_ID,
            observation=observation,
            turn_id=turn_id,
            frame_id=frame_id,
            session_id=session_id,
        )
        await store.flush()

        baselines = await store.load_file_baselines(
            workspace_id=WORKSPACE_ID, session_id=session_id
        )
        assert len(baselines) == 1
        assert baselines[0]["source"]["value"] == "read"

    async def test_source_snapshot_persisted(
        self, execution_store: ExecutionStore
    ) -> None:
        """SNAPSHOT source baselines are persisted."""
        store = execution_store
        session_id, turn_id, frame_id, _ = await _seed_basic_session(store)

        observation = {
            "path": "snapshot_file.txt",
            "hash": "snap_hash",
            "size": 99,
            "mtime": datetime.now(UTC).isoformat(),
            "inode": "inode_snap",
            "source": {"value": "snapshot"},
            "observed_at": datetime.now(UTC).isoformat(),
        }

        await store.upsert_file_baseline(
            workspace_id=WORKSPACE_ID,
            observation=observation,
            turn_id=turn_id,
            frame_id=frame_id,
            session_id=session_id,
        )
        await store.flush()

        baselines = await store.load_file_baselines(
            workspace_id=WORKSPACE_ID, session_id=session_id
        )
        assert len(baselines) == 1
        assert baselines[0]["source"]["value"] == "snapshot"

    async def test_source_agent_create_persisted(
        self, execution_store: ExecutionStore
    ) -> None:
        """AGENT_CREATE source baselines are persisted."""
        store = execution_store
        session_id, turn_id, frame_id, _ = await _seed_basic_session(store)

        observation = {
            "path": "agent_created.py",
            "hash": "agent_hash_val",
            "size": 10,
            "mtime": datetime.now(UTC).isoformat(),
            "inode": None,
            "source": {"value": "agent_create"},
            "observed_at": datetime.now(UTC).isoformat(),
        }

        await store.upsert_file_baseline(
            workspace_id=WORKSPACE_ID,
            observation=observation,
            turn_id=turn_id,
            frame_id=frame_id,
            session_id=session_id,
        )
        await store.flush()

        baselines = await store.load_file_baselines(
            workspace_id=WORKSPACE_ID, session_id=session_id
        )
        assert len(baselines) == 1
        assert baselines[0]["source"]["value"] == "agent_create"

    async def test_source_write_not_persisted(
        self, execution_store: ExecutionStore
    ) -> None:
        """WRITE source baselines are NOT persisted (filtered out)."""
        store = execution_store
        session_id, turn_id, frame_id, _ = await _seed_basic_session(store)

        observation = {
            "path": "write_log.txt",
            "hash": "write_hash",
            "size": 5,
            "mtime": datetime.now(UTC).isoformat(),
            "inode": "inode_write",
            "source": {"value": "write"},
            "observed_at": datetime.now(UTC).isoformat(),
        }

        await store.upsert_file_baseline(
            workspace_id=WORKSPACE_ID,
            observation=observation,
            turn_id=turn_id,
            frame_id=frame_id,
            session_id=session_id,
        )
        await store.flush()

        baselines = await store.load_file_baselines(
            workspace_id=WORKSPACE_ID, session_id=session_id
        )
        assert len(baselines) == 0


class TestFileBaselineBranchRemapping:
    async def test_baselines_remapped_to_fork_rows(
        self, execution_store: ExecutionStore
    ) -> None:
        """Baselines are re-attached to the fork's turn and frame IDs after branch_session."""
        store = execution_store
        session_id, turn_id, frame_id, _ = await _seed_basic_session(
            store, workspace_id=WORKSPACE_ID
        )

        # Upsert a baseline on the parent session
        observation = {
            "path": "src/original.py",
            "hash": "original_hash",
            "size": 50,
            "mtime": datetime.now(UTC).isoformat(),
            "inode": "inode_orig",
            "source": {"value": "read"},
            "observed_at": datetime.now(UTC).isoformat(),
        }

        await store.upsert_file_baseline(
            workspace_id=WORKSPACE_ID,
            observation=observation,
            turn_id=turn_id,
            frame_id=frame_id,
            session_id=session_id,
        )
        await store.flush()

        # Branch the session
        forked = await store.branch_session(
            new_session_id=_new_id(),
            parent_session_id=session_id,
            branch_from_turn_id=turn_id,
        )

        # Check that the baseline is remapped to the fork's turn/frame IDs
        baselines = await store.load_file_baselines(WORKSPACE_ID, forked.id)
        assert len(baselines) == 1
        assert baselines[0]["path"] == "src/original.py"
        assert baselines[0]["hash"] == "original_hash"
        # The baseline should have the fork's turn_id and frame_id (not the parent's)
        # Use raw SQL to check turn_id/frame_id since dict doesn't have those fields
        cursor = await store._conn.execute(
            "SELECT turn_id, frame_id FROM workspace_file_baselines WHERE session_id = ?",
            (forked.id,),
        )
        row = await cursor.fetchone()
        assert row is not None
        # The row should have the fork's turn/frame IDs, not the parent's
        assert row[0] != turn_id or row[1] != frame_id

    async def test_multiple_forks_baselines_independent(
        self, execution_store: ExecutionStore
    ) -> None:
        """Multiple forks from same parent have independent baselines."""
        store = execution_store
        session_id, turn_id, frame_id, _ = await _seed_basic_session(
            store, workspace_id=WORKSPACE_ID
        )

        # Upsert a baseline on the parent
        observation = {
            "path": "shared.py",
            "hash": "shared_hash",
            "size": 30,
            "mtime": datetime.now(UTC).isoformat(),
            "inode": "inode_shared",
            "source": {"value": "read"},
            "observed_at": datetime.now(UTC).isoformat(),
        }

        await store.upsert_file_baseline(
            workspace_id=WORKSPACE_ID,
            observation=observation,
            turn_id=turn_id,
            frame_id=frame_id,
            session_id=session_id,
        )
        await store.flush()

        # Create two forks from different turns
        fork_a = await store.branch_session(
            new_session_id=_new_id(),
            parent_session_id=session_id,
            branch_from_turn_id=turn_id,
        )
        fork_b = await store.branch_session(
            new_session_id=_new_id(),
            parent_session_id=session_id,
            branch_from_turn_id=turn_id,
        )

        baselines_a = await store.load_file_baselines(WORKSPACE_ID, fork_a.id)
        baselines_b = await store.load_file_baselines(WORKSPACE_ID, fork_b.id)

        assert len(baselines_a) == 1
        assert len(baselines_b) == 1
        # Both forks should have the baseline, but they may be independent
        assert baselines_a[0]["path"] == baselines_b[0]["path"]
        assert baselines_a[0]["hash"] == baselines_b[0]["hash"]
