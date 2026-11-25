"""Fixtures for storage integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from local_coding_assistant.storage.execution_store import ExecutionStore


@pytest.fixture
async def execution_store(tmp_path: Path) -> ExecutionStore:
    """Create an ExecutionStore backed by a temporary SQLite file.

    Initializes the store and closes it after the test, flushing any pending
    adapter writes.
    """
    store = ExecutionStore(tmp_path / "test.db")
    await store.init()
    yield store
    await store.close()
