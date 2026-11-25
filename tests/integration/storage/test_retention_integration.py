"""Integration tests for ExecutionStore.cleanup_retention.

Covers per-table TTL cleanup, cascading deletes from sessions,
running-session preservation, tools/templates retention, and
no-error behavior on an empty database.
"""

from __future__ import annotations

from datetime import UTC, datetime

from local_coding_assistant.runtime.events.contracts import EventType
from local_coding_assistant.storage.execution_store import ExecutionStore


def _now_ms() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)


def _older_ms(days: int) -> int:
    return int((datetime.now(UTC).timestamp() - days * 86400) * 1000)


async def _set_sessions_updated_at(
    store: ExecutionStore, session_id: str, ms: int
) -> None:
    await store._conn.execute(
        "UPDATE sessions SET updated_at = ? WHERE id = ?", (ms, session_id)
    )
    await store._conn.commit()


async def _set_tools_created_at(store: ExecutionStore, name: str, ms: int) -> None:
    await store._conn.execute(
        "UPDATE tools SET created_at = ? WHERE name = ?", (ms, name)
    )
    await store._conn.commit()


async def _set_templates_created_at(store: ExecutionStore, path: str, ms: int) -> None:
    await store._conn.execute(
        "UPDATE templates SET created_at = ? WHERE path = ?", (ms, path)
    )
    await store._conn.commit()


async def _set_repo_snapshots_created_at(
    store: ExecutionStore, repo_snapshot_hash: str, ms: int
) -> None:
    await store._conn.execute(
        "UPDATE repo_snapshots SET created_at = ? WHERE repo_snapshot_hash = ?",
        (ms, repo_snapshot_hash),
    )
    await store._conn.commit()


async def _set_workspaces_updated_at(
    store: ExecutionStore, workspace_id: str, ms: int
) -> None:
    await store._conn.execute(
        "UPDATE workspaces SET updated_at = ? WHERE id = ?", (ms, workspace_id)
    )
    await store._conn.commit()


async def _create_session_graph(
    store: ExecutionStore,
    session_id: str,
    turn_id: str,
    frame_id: str,
    snapshot_id: str,
    status: str = "idle",
) -> None:
    """Create a session with a turn, frame, snapshot, event, and message."""
    await store.create_session(
        session_id=session_id,
        user_input="retention query",
        agent_mode="frame_agent",
        model_name="gpt-4",
    )
    if status != "running":
        await store.session_repo.update_status(session_id, status)
        await store._conn.commit()
    await store.create_turn(
        turn_id=turn_id,
        session_id=session_id,
        trace_id="trace_1",
        turn_index=0,
        user_input="retention query",
        status=status,
        context_snapshot_id="snap_1",
        config_snapshot_json="{}",
    )
    await store.create_context_snapshot(
        snapshot_id=snapshot_id,
        turn_id=turn_id,
        content_hash="snap_hash",
    )
    await store.create_frame(
        frame_id=frame_id,
        session_id=session_id,
        turn_id=turn_id,
        trace_id="trace_1",
        iteration=0,
    )
    await store.append_event(
        trace_id="trace_1",
        session_id=session_id,
        turn_id=turn_id,
        frame_id=frame_id,
        seq=1,
        event_type=EventType.FRAME_START,
        data_json="{}",
    )
    await store.append_message(
        session_id=session_id,
        turn_id=turn_id,
        frame_id=frame_id,
        seq=1,
        role="user",
        content="retention query",
    )


class TestRetentionIntegration:
    async def test_cleanup_retention_deletes_expired_rows_and_cascade(
        self, execution_store: ExecutionStore
    ) -> None:
        """Expired sessions, tools, templates, repo snapshots, workspaces deleted."""
        store = execution_store
        default_days = 1
        old_ms = _older_ms(2)  # well before cutoff
        new_ms = _now_ms()  # after cutoff

        # ---- sessions (deleted -> cascades to turn/frame/snapshot/event/message) ----
        await _create_session_graph(
            store, "sess_old", "turn_old", "frame_old", "snap_old", status="idle"
        )
        await _set_sessions_updated_at(store, "sess_old", old_ms)

        # recent session preserved
        await _create_session_graph(
            store,
            "sess_recent",
            "turn_recent",
            "frame_recent",
            "snap_recent",
            status="idle",
        )
        await _set_sessions_updated_at(store, "sess_recent", new_ms)

        # old running session preserved (running sessions are protected)
        await _create_session_graph(
            store,
            "sess_old_running",
            "turn_run",
            "frame_run",
            "snap_run",
            status="running",
        )
        await _set_sessions_updated_at(store, "sess_old_running", old_ms)

        # ---- tools (deleted by same default TTL) ----
        await store.register_tool("tool_old", "{}", "low")
        await _set_tools_created_at(store, "tool_old", old_ms)
        await store.register_tool("tool_new", "{}", "low")
        await _set_tools_created_at(store, "tool_new", new_ms)

        # ---- templates ----
        await store.register_template("/path/old.j2", "content", "hash_old", "1.0.0")
        await _set_templates_created_at(store, "/path/old.j2", old_ms)
        await store.register_template("/path/new.j2", "content", "hash_new", "1.0.0")
        await _set_templates_created_at(store, "/path/new.j2", new_ms)

        # ---- repo snapshots ----
        await store.register_repo_snapshot(
            "/root/old", '{"a":1}', None, "hash_old_repo"
        )
        await _set_repo_snapshots_created_at(store, "hash_old_repo", old_ms)
        await store.register_repo_snapshot(
            "/root/new", '{"a":1}', None, "hash_new_repo"
        )
        await _set_repo_snapshots_created_at(store, "hash_new_repo", new_ms)

        # ---- workspaces ----
        await store.upsert_workspace("ws_old", "/ws/old", "cfg_old")
        await _set_workspaces_updated_at(store, "ws_old", old_ms)
        await store.upsert_workspace("ws_new", "/ws/new", "cfg_new")
        await _set_workspaces_updated_at(store, "ws_new", new_ms)

        await store.cleanup_retention({"default": default_days})

        # sessions + cascade removed
        assert await store.get_session("sess_old") is None
        assert await store.get_turn("turn_old") is None
        assert await store.frame_repo.get("frame_old") is None
        assert await store.get_context_snapshot("snap_old") is None
        assert await store.get_frame_events("frame_old") == []
        assert await store.get_all_messages("sess_old") == []

        # recent session preserved
        assert await store.get_session("sess_recent") is not None
        assert await store.get_turn("turn_recent") is not None

        # old running session preserved
        assert await store.get_session("sess_old_running") is not None

        # tools/templates/repo snapshots/workspaces aged out
        assert await store.tool_repo.get_by_name("tool_old") is None
        assert await store.tool_repo.get_by_name("tool_new") is not None
        assert await store.template_repo.get_by_path("/path/old.j2") is None
        assert await store.template_repo.get_by_path("/path/new.j2") is not None
        assert await store.repo_snapshot_repo.get_by_hash("hash_old_repo") is None
        assert await store.repo_snapshot_repo.get_by_hash("hash_new_repo") is not None
        assert await store.workspace_repo.get_by_root("/ws/old") is None
        assert await store.workspace_repo.get_by_root("/ws/new") is not None

    async def test_cleanup_retention_empty_db_no_error(
        self, execution_store: ExecutionStore
    ) -> None:
        """cleanup_retention on an empty database raises no error."""
        store = execution_store
        await store.cleanup_retention({"default": 30})
        await store.cleanup_retention({})
        await store.cleanup_retention({"default": 0, "sessions": 7, "tools": 365})

    async def test_cleanup_retention_running_session_protected(
        self, execution_store: ExecutionStore
    ) -> None:
        """A running session is never deleted regardless of age."""
        store = execution_store
        old_ms = _older_ms(10)

        await _create_session_graph(
            store, "sess_run", "turn_run", "frame_run", "snap_run2", status="running"
        )
        await _set_sessions_updated_at(store, "sess_run", old_ms)

        await store.cleanup_retention({"default": 1})

        assert await store.get_session("sess_run") is not None
        assert await store.get_turn("turn_run") is not None
        assert await store.frame_repo.get("frame_run") is not None

    async def test_cleanup_retention_preserves_recent_everything(
        self, execution_store: ExecutionStore
    ) -> None:
        """Everything newer than the TTL is preserved."""
        store = execution_store
        new_ms = _now_ms()

        await _create_session_graph(
            store, "sess_new", "turn_new", "frame_new", "snap_new", status="idle"
        )
        await _set_sessions_updated_at(store, "sess_new", new_ms)

        await store.register_tool("tool_recent", "{}", "low")
        await _set_tools_created_at(store, "tool_recent", new_ms)

        await store.register_template("/path/recent.j2", "content", "hash_r", "1.0.0")
        await _set_templates_created_at(store, "/path/recent.j2", new_ms)

        await store.register_repo_snapshot(
            "/root/recent", '{"a":1}', None, "hash_recent_repo"
        )
        await _set_repo_snapshots_created_at(store, "hash_recent_repo", new_ms)

        await store.upsert_workspace("ws_recent", "/ws/recent", "cfg_r")
        await _set_workspaces_updated_at(store, "ws_recent", new_ms)

        await store.cleanup_retention({"default": 1})

        assert await store.get_session("sess_new") is not None
        assert await store.get_turn("turn_new") is not None
        assert await store.frame_repo.get("frame_new") is not None
        assert await store.tool_repo.get_by_name("tool_recent") is not None
        assert await store.template_repo.get_by_path("/path/recent.j2") is not None
        assert (
            await store.repo_snapshot_repo.get_by_hash("hash_recent_repo") is not None
        )
        assert await store.workspace_repo.get_by_root("/ws/recent") is not None
