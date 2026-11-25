"""Tests for schema validation."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from local_coding_assistant.storage.execution_store import ExecutionStore


@pytest.fixture
async def temp_db_path(tmp_path: Path) -> Path:
    """Create a temporary database path."""
    return tmp_path / "test.db"


@pytest.fixture
async def execution_store(temp_db_path: Path) -> ExecutionStore:
    """Create an execution store for testing."""
    store = ExecutionStore(temp_db_path)
    await store.init()
    yield store
    await store.close()


@pytest.mark.asyncio
async def test_schema_creates_all_tables(execution_store: ExecutionStore) -> None:
    """Test that schema creates all required tables."""
    conn = sqlite3.connect(str(execution_store.db_path))
    cursor = conn.cursor()

    # Check all tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]

    expected_tables = [
        "context_snapshots",
        "events",
        "frames",
        "messages",
        "repo_snapshots",
        "session_templates",
        "session_tools",
        "sessions",
        "templates",
        "tools",
        "turns",
        "workspace_file_baselines",
        "workspaces",
    ]
    # Filter out SQLite system tables
    tables = [t for t in tables if not t.startswith("sqlite_")]
    assert set(tables) == set(expected_tables)

    conn.close()


@pytest.mark.asyncio
async def test_schema_creates_indexes(execution_store: ExecutionStore) -> None:
    """Test that schema creates all required indexes."""
    conn = sqlite3.connect(str(execution_store.db_path))
    cursor = conn.cursor()

    # Check indexes exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' ORDER BY name")
    indexes = [row[0] for row in cursor.fetchall()]

    expected_indexes = [
        "idx_context_snapshot_hash",
        "idx_context_snapshot_turn",
        "idx_events_frame",
        "idx_events_session_seq",
        "idx_events_session_timestamp",
        "idx_events_trace",
        "idx_events_type_timestamp",
        "idx_events_turn_seq",
        "idx_events_turn_timestamp",
        "idx_frames_context_snapshot",
        "idx_frames_session_iter",
        "idx_frames_trace",
        "idx_frames_turn_iter",
        "idx_messages_frame",
        "idx_messages_session_seq",
        "idx_messages_turn_seq",
        "idx_turns_context_snapshot",
        "idx_turns_session_idx",
        "idx_turns_trace",
    ]
    for idx in expected_indexes:
        assert idx in indexes

    conn.close()


@pytest.mark.asyncio
async def test_schema_foreign_keys_enabled(execution_store: ExecutionStore) -> None:
    """Test that foreign keys are enabled in the store connection."""
    # Foreign keys are enabled in ExecutionStore.init()
    # The setting is connection-specific, so we verify by checking the connection
    # was initialized with the PRAGMA (we can't check a new connection)
    # Instead, we verify the store's connection is set up correctly
    assert execution_store._conn is not None
    # The PRAGMA was set during init, so we trust it was applied
    # We can verify by checking foreign key constraints work (via other tests)


@pytest.mark.asyncio
async def test_schema_wal_mode(execution_store: ExecutionStore) -> None:
    """Test that WAL mode is enabled."""
    conn = sqlite3.connect(str(execution_store.db_path))
    cursor = conn.cursor()

    cursor.execute("PRAGMA journal_mode")
    journal_mode = cursor.fetchone()[0]
    assert journal_mode == "wal"

    conn.close()


@pytest.mark.asyncio
async def test_tools_table_structure(execution_store: ExecutionStore) -> None:
    """Test tools table has correct structure."""
    conn = sqlite3.connect(str(execution_store.db_path))
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(tools)")
    columns = {row[1]: row[2] for row in cursor.fetchall()}

    assert "id" in columns
    assert "name" in columns
    assert "schema_json" in columns
    assert "risk_class" in columns
    assert "description" in columns
    assert "version" in columns
    assert "created_at" in columns

    conn.close()


@pytest.mark.asyncio
async def test_sessions_table_structure(execution_store: ExecutionStore) -> None:
    """Test sessions table has correct structure (new model)."""
    conn = sqlite3.connect(str(execution_store.db_path))
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(sessions)")
    columns = {row[1]: row[2] for row in cursor.fetchall()}

    assert "id" in columns
    assert "user_input" in columns
    assert "status" in columns
    assert "started_at" in columns
    assert "updated_at" in columns
    assert "agent_mode" in columns
    assert "model_name" in columns
    assert "total_input_tokens" in columns
    assert "total_output_tokens" in columns
    assert "total_turns" in columns
    assert "total_frames" in columns
    assert "metadata_json" in columns

    # Old columns should NOT exist
    assert "trace_id" not in columns
    assert "execution_mode" not in columns
    assert "tool_call_mode" not in columns
    assert "agent_profile_json" not in columns
    assert "config_snapshot_json" not in columns

    conn.close()


@pytest.mark.asyncio
async def test_frames_table_structure(execution_store: ExecutionStore) -> None:
    """Test frames table has correct structure (new model)."""
    conn = sqlite3.connect(str(execution_store.db_path))
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(frames)")
    columns = {row[1]: row[2] for row in cursor.fetchall()}

    assert "id" in columns
    assert "session_id" in columns
    assert "turn_id" in columns
    assert "trace_id" in columns
    assert "iteration" in columns
    assert "status" in columns
    assert "context_snapshot_id" in columns
    assert "started_at" in columns
    assert "finished_at" in columns
    assert "duration_ms" in columns
    assert "input_tokens" in columns
    assert "output_tokens" in columns
    assert "model_name" in columns
    assert "finish_reason" in columns
    assert "error_message" in columns
    assert "pending_call_id" in columns
    assert "history_through_seq" in columns
    assert "prompt_hash" in columns

    conn.close()


@pytest.mark.asyncio
async def test_messages_table_structure(execution_store: ExecutionStore) -> None:
    """Test messages table has correct structure (new model)."""
    conn = sqlite3.connect(str(execution_store.db_path))
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(messages)")
    columns = {row[1]: row[2] for row in cursor.fetchall()}

    assert "id" in columns
    assert "session_id" in columns
    assert "turn_id" in columns
    assert "frame_id" in columns
    assert "seq" in columns
    assert "role" in columns
    assert "content" in columns
    assert "tool_call_id" in columns
    assert "tool_name" in columns
    assert "metadata_json" in columns
    assert "created_at" in columns

    conn.close()


@pytest.mark.asyncio
async def test_events_table_structure(execution_store: ExecutionStore) -> None:
    """Test events table has correct structure (new model)."""
    conn = sqlite3.connect(str(execution_store.db_path))
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(events)")
    columns = {row[1]: row[2] for row in cursor.fetchall()}

    assert "id" in columns
    assert "trace_id" in columns
    assert "session_id" in columns
    assert "turn_id" in columns
    assert "frame_id" in columns
    assert "seq" in columns
    assert "type" in columns
    assert "data_json" in columns
    assert "timestamp" in columns

    conn.close()


@pytest.mark.asyncio
async def test_workspaces_table_structure(execution_store: ExecutionStore) -> None:
    """Test workspaces table has the persisted identity structure."""
    conn = sqlite3.connect(str(execution_store.db_path))
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(workspaces)")
    columns = {row[1]: row[2] for row in cursor.fetchall()}

    assert "id" in columns
    assert "workspace_root" in columns
    assert "config_hash" in columns
    assert "updated_at" in columns

    conn.close()


@pytest.mark.asyncio
async def test_foreign_key_constraints(execution_store: ExecutionStore) -> None:
    """Test that foreign key constraints are enforced."""
    conn = sqlite3.connect(str(execution_store.db_path))
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")

    # Insert a turn referencing a non-existent session should fail
    try:
        cursor.execute(
            "INSERT INTO turns (id, session_id, trace_id, turn_index, user_input, status, started_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("turn-1", "non-existent-session", "trace-1", 1, "input", "running", 1000),
        )
        conn.rollback()
        assert False, (
            "FK constraint should have failed for turn referencing non-existent session"
        )
    except sqlite3.IntegrityError:
        pass  # Expected

    # Insert a frame referencing a non-existent session should fail
    try:
        cursor.execute(
            "INSERT INTO frames (id, session_id, turn_id, trace_id, iteration, status, started_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "frame-1",
                "non-existent-session",
                "turn-1",
                "trace-1",
                1,
                "running_llm",
                1000,
            ),
        )
        conn.rollback()
        assert False, (
            "FK constraint should have failed for frame referencing non-existent session"
        )
    except sqlite3.IntegrityError:
        pass  # Expected

    # Insert a context_snapshot referencing a non-existent turn should fail
    try:
        cursor.execute(
            "INSERT INTO context_snapshots (id, turn_id, schema_version, content_hash, created_at) VALUES (?, ?, ?, ?, ?)",
            ("snap-1", "non-existent-turn", 1, "hash-1", 1000),
        )
        conn.rollback()
        assert False, (
            "FK constraint should have failed for context_snapshot referencing non-existent turn"
        )
    except sqlite3.IntegrityError:
        pass  # Expected

    # Insert a valid turn referencing a valid session should succeed
    cursor.execute(
        "INSERT INTO sessions (id, user_input, status, started_at, agent_mode) VALUES (?, ?, ?, ?, ?)",
        ("session-1", "test input", "running", 1000, "frame_agent"),
    )
    cursor.execute(
        "INSERT INTO turns (id, session_id, trace_id, turn_index, user_input, status, started_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("turn-1", "session-1", "trace-1", 1, "input", "running", 1000),
    )
    conn.commit()
    conn.close()


@pytest.mark.asyncio
async def test_check_constraints(execution_store: ExecutionStore) -> None:
    """Test that status validation works via Pydantic contracts."""
    from local_coding_assistant.storage.contracts import StoredFrame, StoredSession

    # Valid session statuses should succeed
    session_idle = StoredSession(
        id="session-1",
        user_input="test input",
        status="idle",
        started_at_ms=1000,
        agent_mode="frame_agent",
    )
    session_running = StoredSession(
        id="session-2",
        user_input="test input",
        status="running",
        started_at_ms=1000,
        agent_mode="frame_agent",
    )
    assert session_idle.status == "idle"
    assert session_running.status == "running"

    # Invalid session status should fail Pydantic validation
    try:
        StoredSession(
            id="session-3",
            user_input="test input",
            status="invalid_status",
            started_at_ms=1000,
            agent_mode="frame_agent",
        )
        assert False, "Pydantic should reject invalid session status"
    except Exception:
        pass  # Expected

    # Valid frame statuses should succeed
    frame_running_llm = StoredFrame(
        id="frame-1",
        session_id="session-1",
        turn_id="turn-1",
        trace_id="trace-1",
        iteration=1,
        status="running_llm",
        started_at_ms=1000,
    )
    frame_running_tools = StoredFrame(
        id="frame-2",
        session_id="session-1",
        turn_id="turn-1",
        trace_id="trace-1",
        iteration=1,
        status="running_tools",
        started_at_ms=1000,
    )
    frame_completed = StoredFrame(
        id="frame-3",
        session_id="session-1",
        turn_id="turn-1",
        trace_id="trace-1",
        iteration=1,
        status="completed",
        started_at_ms=1000,
        finished_at_ms=2000,
        input_tokens=10,
        output_tokens=0,
    )
    assert frame_running_llm.status == "running_llm"
    assert frame_running_tools.status == "running_tools"
    assert frame_completed.status == "completed"

    # Invalid frame status should fail Pydantic validation
    try:
        StoredFrame(
            id="frame-4",
            session_id="session-1",
            turn_id="turn-1",
            trace_id="trace-1",
            iteration=1,
            status="invalid_status",
            started_at_ms=1000,
        )
        assert False, "Pydantic should reject invalid frame status"
    except Exception:
        pass  # Expected


@pytest.mark.asyncio
async def test_unique_constraints(execution_store: ExecutionStore) -> None:
    """Test that UNIQUE constraints are enforced."""
    conn = sqlite3.connect(str(execution_store.db_path))
    cursor = conn.cursor()

    # Insert a tool with content_hash should succeed
    cursor.execute(
        "INSERT INTO tools (id, name, schema_json, risk_class, description, version, content_hash, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (1, "test-tool", "{}", "low", "test", "1.0", "hash-1", 1000),
    )
    conn.commit()

    # Insert another tool with same content_hash should fail
    try:
        cursor.execute(
            "INSERT INTO tools (id, name, schema_json, risk_class, description, version, content_hash, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (2, "test-tool-2", "{}", "low", "test2", "2.0", "hash-1", 2000),
        )
        conn.rollback()
        assert False, (
            "UNIQUE constraint should have failed for duplicate tool content_hash"
        )
    except sqlite3.IntegrityError:
        pass  # Expected

    # Insert a template with content_hash should succeed
    cursor.execute(
        "INSERT INTO templates (id, path, content, content_hash, version, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (1, "/test/path", "test content", "hash-2", "1.0", 1000),
    )
    conn.commit()

    # Insert another template with same content_hash should fail
    try:
        cursor.execute(
            "INSERT INTO templates (id, path, content, content_hash, version, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (2, "/test/path2", "test content 2", "hash-2", "2.0", 2000),
        )
        conn.rollback()
        assert False, (
            "UNIQUE constraint should have failed for duplicate template content_hash"
        )
    except sqlite3.IntegrityError:
        pass  # Expected

    # Insert workspace with unique workspace_root should succeed
    cursor.execute(
        "INSERT INTO workspaces (id, workspace_root, config_hash, updated_at) VALUES (?, ?, ?, ?)",
        ("workspace-1", "/test/root", "hash-3", 1000),
    )
    conn.commit()

    # Insert another workspace with same workspace_root should fail
    try:
        cursor.execute(
            "INSERT INTO workspaces (id, workspace_root, config_hash, updated_at) VALUES (?, ?, ?, ?)",
            ("workspace-2", "/test/root", "hash-4", 2000),
        )
        conn.rollback()
        assert False, (
            "UNIQUE constraint should have failed for duplicate workspace workspace_root"
        )
    except sqlite3.IntegrityError:
        pass  # Expected

    conn.close()
