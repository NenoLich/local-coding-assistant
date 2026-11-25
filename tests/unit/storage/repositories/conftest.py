"""Fixtures for repository unit tests."""

from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest

from local_coding_assistant.storage.repositories import (
    ContextSnapshotRepo,
    EventRepo,
    FrameRepo,
    MessageRepo,
    RepoSnapshotRepo,
    SessionRepo,
    TemplateRepo,
    ToolRepo,
    TurnRepo,
    WorkspaceFileBaselineRepo,
    WorkspaceRepo,
)


@pytest.fixture
async def db_conn(tmp_path: Path):
    """Create an in-memory SQLite connection with schema."""
    conn = await aiosqlite.connect(":memory:")
    await conn.execute("PRAGMA foreign_keys=ON")

    # Load and execute schema
    schema_path = (
        Path(__file__).parents[4]
        / "src"
        / "local_coding_assistant"
        / "storage"
        / "schema.sql"
    )
    schema_sql = schema_path.read_text()
    await conn.executescript(schema_sql)
    await conn.commit()

    yield conn

    await conn.close()


@pytest.fixture
def session_repo(db_conn: aiosqlite.Connection):
    """SessionRepo fixture."""
    return SessionRepo(db_conn)


@pytest.fixture
def turn_repo(db_conn: aiosqlite.Connection):
    """TurnRepo fixture."""
    return TurnRepo(db_conn)


@pytest.fixture
def frame_repo(db_conn: aiosqlite.Connection):
    """FrameRepo fixture."""
    return FrameRepo(db_conn)


@pytest.fixture
def message_repo(db_conn: aiosqlite.Connection):
    """MessageRepo fixture."""
    return MessageRepo(db_conn)


@pytest.fixture
def event_repo(db_conn: aiosqlite.Connection):
    """EventRepo fixture."""
    return EventRepo(db_conn)


@pytest.fixture
def context_snapshot_repo(db_conn: aiosqlite.Connection):
    """ContextSnapshotRepo fixture."""
    return ContextSnapshotRepo(db_conn)


@pytest.fixture
def tool_repo(db_conn: aiosqlite.Connection):
    """ToolRepo fixture."""
    return ToolRepo(db_conn)


@pytest.fixture
def template_repo(db_conn: aiosqlite.Connection):
    """TemplateRepo fixture."""
    return TemplateRepo(db_conn)


@pytest.fixture
def repo_snapshot_repo(db_conn: aiosqlite.Connection):
    """RepoSnapshotRepo fixture."""
    return RepoSnapshotRepo(db_conn)


@pytest.fixture
def workspace_repo(db_conn: aiosqlite.Connection):
    """WorkspaceRepo fixture."""
    return WorkspaceRepo(db_conn)


@pytest.fixture
def workspace_file_baseline_repo(db_conn: aiosqlite.Connection):
    """WorkspaceFileBaselineRepo fixture."""
    return WorkspaceFileBaselineRepo(db_conn)
