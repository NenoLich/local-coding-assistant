"""Integration tests for ExecutionStore workspace identity.

Covers workspace record lifecycle (upsert/get/refresh config hash),
config drift detection, multiple sessions sharing a workspace, and
file baselines scoped to workspace_id (with optional session filter).
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime

from local_coding_assistant.storage.execution_store import ExecutionStore


def _new_id() -> str:
    return f"ws_{uuid.uuid4().hex[:12]}"


def _observation(
    path: str = "src/a.py",
    hash_val: str = "hash_a",
    size: int = 10,
    source: str = "read",
) -> dict[str, any]:
    return {
        "path": path,
        "hash": hash_val,
        "size": size,
        "mtime": datetime.now(UTC).isoformat(),
        "inode": None,
        "observed_at": datetime.now(UTC).isoformat(),
        "source": {"value": source},
    }


async def _create_session(store: ExecutionStore, session_id: str) -> None:
    await store.create_session(
        session_id=session_id,
        user_input="hello",
        agent_mode="frame_agent",
        model_name="gpt-4",
    )


class TestWorkspaceUpsert:
    async def test_upsert_workspace_creates_record(
        self, execution_store: ExecutionStore
    ) -> None:
        store = execution_store
        workspace_id = _new_id()
        root = "/root/workspace"
        config_hash = "config_hash_1"

        record = await store.upsert_workspace(workspace_id, root, config_hash)

        assert record.id == workspace_id
        assert record.workspace_root == root
        assert record.config_hash == config_hash
        assert record.updated_at > 0

        fetched = await store.get_workspace(root)
        assert fetched is not None
        assert fetched.id == workspace_id
        assert fetched.config_hash == config_hash

    async def test_upsert_workspace_same_root_shares_identity(
        self, execution_store: ExecutionStore
    ) -> None:
        store = execution_store
        root = "/root/workspace"

        ws_id_1 = _new_id()
        ws_id_2 = _new_id()

        await store.upsert_workspace(ws_id_1, root, "hash_a")
        await store.upsert_workspace(ws_id_2, root, "hash_b")

        record = await store.get_workspace(root)
        assert record is not None
        assert record.id == ws_id_1
        assert record.config_hash == "hash_a"

    async def test_upsert_workspace_different_roots_get_distinct_ids(
        self, execution_store: ExecutionStore
    ) -> None:
        store = execution_store

        record_a = await store.upsert_workspace(_new_id(), "/root/a", "hash_a")
        record_b = await store.upsert_workspace(_new_id(), "/root/b", "hash_b")

        assert record_a.id != record_b.id
        assert (await store.get_workspace("/root/a")) is not None
        assert (await store.get_workspace("/root/b")) is not None


class TestWorkspaceGet:
    async def test_get_workspace_returns_none_when_absent(
        self, execution_store: ExecutionStore
    ) -> None:
        store = execution_store
        assert await store.get_workspace("/root/nonexistent") is None

    async def test_get_workspace_finds_by_root_and_touches_updated_at(
        self, execution_store: ExecutionStore
    ) -> None:
        store = execution_store
        workspace_id = _new_id()
        root = "/root/workspace"

        await store.upsert_workspace(workspace_id, root, "hash_a")

        before = await store.get_workspace(root)
        assert before is not None
        assert before.id == workspace_id
        assert before.config_hash == "hash_a"

        await asyncio.sleep(0.01)
        after = await store.get_workspace(root)

        assert after is not None
        assert after.id == workspace_id
        assert after.updated_at >= before.updated_at


class TestWorkspaceRefreshConfigHash:
    async def test_refresh_workspace_config_hash_detects_drift(
        self, execution_store: ExecutionStore
    ) -> None:
        store = execution_store
        workspace_id = _new_id()
        root = "/root/workspace"

        await store.upsert_workspace(workspace_id, root, "hash_a")

        before = await store.get_workspace(root)
        assert before.config_hash == "hash_a"

        await asyncio.sleep(0.01)
        await store.refresh_workspace_config_hash(workspace_id, "hash_b")

        after = await store.get_workspace(root)
        assert after is not None
        assert after.config_hash == "hash_b"
        assert after.updated_at >= before.updated_at

    async def test_refresh_workspace_config_hash_nonexistent_workspace_no_error(
        self, execution_store: ExecutionStore
    ) -> None:
        store = execution_store
        await store.refresh_workspace_config_hash(_new_id(), "hash_b")


class TestWorkspaceSharedBySessions:
    async def test_multiple_sessions_share_workspace_id(
        self, execution_store: ExecutionStore
    ) -> None:
        store = execution_store
        root = "/root/workspace"
        workspace_id = _new_id()

        await store.upsert_workspace(workspace_id, root, "hash_a")

        session_a = _new_id()
        session_b = _new_id()
        await _create_session(store, session_a)
        await _create_session(store, session_b)

        record = await store.get_workspace(root)
        assert record is not None
        assert record.id == workspace_id

    async def test_file_baselines_scoped_to_workspace_and_session(
        self, execution_store: ExecutionStore
    ) -> None:
        store = execution_store
        root = "/root/workspace"
        workspace_id = _new_id()
        await store.upsert_workspace(workspace_id, root, "hash_a")

        session_a = _new_id()
        session_b = _new_id()
        turn_a = _new_id()
        turn_b = _new_id()
        await _create_session(store, session_a)
        await _create_session(store, session_b)
        await store.create_turn(
            turn_id=turn_a,
            session_id=session_a,
            trace_id=_new_id(),
            turn_index=0,
            user_input="first",
        )
        await store.create_turn(
            turn_id=turn_b,
            session_id=session_b,
            trace_id=_new_id(),
            turn_index=0,
            user_input="second",
        )

        await store.upsert_file_baseline(
            workspace_id=workspace_id,
            observation=_observation(path="src/a.py", hash_val="hash_a"),
            turn_id=turn_a,
            frame_id=None,
            session_id=session_a,
        )
        await store.upsert_file_baseline(
            workspace_id=workspace_id,
            observation=_observation(path="src/b.py", hash_val="hash_b"),
            turn_id=turn_b,
            frame_id=None,
            session_id=session_b,
        )

        all_baselines = await store.load_file_baselines(workspace_id)
        assert {b["path"] for b in all_baselines} == {"src/a.py", "src/b.py"}

        session_a_baselines = await store.load_file_baselines(workspace_id, session_a)
        assert [b["path"] for b in session_a_baselines] == ["src/a.py"]

        session_b_baselines = await store.load_file_baselines(workspace_id, session_b)
        assert [b["path"] for b in session_b_baselines] == ["src/b.py"]

    async def test_workspace_persists_across_close_reopen(
        self, execution_store: ExecutionStore
    ) -> None:
        store1 = execution_store
        db_path = store1.db_path

        workspace_id = _new_id()
        root = "/root/workspace"
        await store1.upsert_workspace(workspace_id, root, "hash_a")
        await store1.close()

        store2 = ExecutionStore(db_path)
        await store2.init()
        try:
            record = await store2.get_workspace(root)
            assert record is not None
            assert record.id == workspace_id
            assert record.config_hash == "hash_a"
        finally:
            await store2.close()
