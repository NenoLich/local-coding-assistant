"""Fixtures for StoreEventAdapter unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from local_coding_assistant.runtime.events.contracts import ApprovalRequestForm
from local_coding_assistant.storage.execution_store import ExecutionStore
from local_coding_assistant.storage.store_event_adapter import StoreEventAdapter


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


@pytest.fixture
def adapter() -> StoreEventAdapter:
    """Create a fresh StoreEventAdapter for each test."""
    return StoreEventAdapter()


def approval_requests(*, count: int = 1) -> list[ApprovalRequestForm]:
    """Build approval request forms for ApprovalRequiredEvent."""
    return [
        ApprovalRequestForm(
            approval_token=f"token_{i}",
            tool_call_id=f"call_{i}",
            tool_name="write_file",
            arguments={"path": f"/tmp/file_{i}.txt"},  # noqa: S108 - test fixture path, not a real temp dir
        )
        for i in range(count)
    ]
