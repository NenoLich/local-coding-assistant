"""Unit tests for the simplified scheduler lock system.

Tests effect-based locking with reader-writer semantics:
- Shared locks for reads, exclusive locks for writes
- Capacity semaphores for concurrency limits
- Tool call classification into parallel groups
- Deadlock-free lock ordering via sorted key acquisition
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from local_coding_assistant.tools_platform.contracts import (
    ConcurrencyScope,
    ConflictRule,
    LockLifetime,
    LockMode,
    OperationEffect,
    ToolDefinition,
    ValidatedToolCall,
    compute_command_locks,
    compute_file_create_locks,
    compute_file_delete_locks,
    compute_file_read_locks,
    compute_file_write_locks,
)
from local_coding_assistant.tools_platform.exceptions import ToolConflictError
from local_coding_assistant.tools_platform.scheduler import ReaderWriterLock, Scheduler
from local_coding_assistant.tools_platform.workspace.manager import WorkspaceManager


def _make_call(
    concurrency_scope: ConcurrencyScope, input_args: dict | None = None
) -> ValidatedToolCall:
    """Build a ValidatedToolCall mock with the given concurrency scope."""
    call = MagicMock(spec=ValidatedToolCall)
    call.tool_definition = MagicMock(spec=ToolDefinition)
    call.tool_definition.concurrency_scope = concurrency_scope
    call.input_model = MagicMock()
    call.input_model.model_dump.return_value = input_args or {}
    return call


def _read_scope(path: str) -> ConcurrencyScope:
    return ConcurrencyScope(
        effect=OperationEffect.READ,
        resource_locks=compute_file_read_locks,
        conflict_rule=ConflictRule.WAIT,
        lifetime=LockLifetime.EXECUTION,
    )


def _write_scope(path: str) -> ConcurrencyScope:
    return ConcurrencyScope(
        effect=OperationEffect.WRITE,
        resource_locks=compute_file_write_locks,
        conflict_rule=ConflictRule.REQUIRE_FRESH,
        lifetime=LockLifetime.COMMIT,
    )


def _create_scope(path: str) -> ConcurrencyScope:
    return ConcurrencyScope(
        effect=OperationEffect.WRITE,
        resource_locks=compute_file_create_locks,
        conflict_rule=ConflictRule.REQUIRE_FRESH,
        lifetime=LockLifetime.COMMIT,
    )


def _delete_scope(path: str) -> ConcurrencyScope:
    return ConcurrencyScope(
        effect=OperationEffect.WRITE,
        resource_locks=compute_file_delete_locks,
        conflict_rule=ConflictRule.REQUIRE_FRESH,
        lifetime=LockLifetime.COMMIT,
    )


def _process_scope() -> ConcurrencyScope:
    return ConcurrencyScope(
        effect=OperationEffect.PROCESS,
        resource_locks=compute_command_locks,
        capacity_group="processes",
        conflict_rule=ConflictRule.WAIT,
        lifetime=LockLifetime.DETACHED_PROCESS,
    )


def _capacity_scope(group: str = "processes") -> ConcurrencyScope:
    """Capacity-only scope: takes a semaphore permit with no resource locks."""
    return ConcurrencyScope(
        effect=OperationEffect.PROCESS,
        resource_locks=[],
        capacity_group=group,
        conflict_rule=ConflictRule.WAIT,
        lifetime=LockLifetime.DETACHED_PROCESS,
    )


@pytest.fixture
def workspace_manager(tmp_path):
    manager = MagicMock(spec=WorkspaceManager)
    manager.workspace_root = tmp_path
    return manager


@pytest.fixture
def config_manager():
    config_manager = MagicMock()
    config_manager.global_config.tools_platform.concurrency.max_parallel_reads = 8
    config_manager.global_config.tools_platform.concurrency.max_parallel_processes = 2
    return config_manager


@pytest.fixture
def scheduler(workspace_manager, config_manager):
    return Scheduler(workspace_manager, config_manager)


class TestReaderWriterLock:
    """Test the ReaderWriterLock primitive directly."""

    @pytest.mark.asyncio
    async def test_multiple_concurrent_readers(self):
        lock = ReaderWriterLock()
        reader_count = 5
        active_readers = 0
        max_concurrent = 0

        async def reader():
            nonlocal active_readers, max_concurrent
            await lock.acquire_shared()
            active_readers += 1
            max_concurrent = max(max_concurrent, active_readers)
            await asyncio.sleep(0.01)
            active_readers -= 1
            await lock.release_shared()

        tasks = [asyncio.create_task(reader()) for _ in range(reader_count)]
        await asyncio.gather(*tasks)

        assert max_concurrent == reader_count

    @pytest.mark.asyncio
    async def test_writer_waits_for_readers(self):
        lock = ReaderWriterLock()
        reader_holding = asyncio.Event()
        writer_acquired = asyncio.Event()

        async def reader():
            await lock.acquire_shared()
            reader_holding.set()
            await asyncio.sleep(0.1)
            await lock.release_shared()

        async def writer():
            await lock.acquire_exclusive()
            writer_acquired.set()
            await lock.release_exclusive()

        reader_task = asyncio.create_task(reader())
        await reader_holding.wait()

        writer_task = asyncio.create_task(writer())
        await asyncio.sleep(0.05)
        assert not writer_acquired.is_set()

        await asyncio.gather(reader_task, writer_task)
        assert writer_acquired.is_set()

    @pytest.mark.asyncio
    async def test_readers_wait_for_writer(self):
        lock = ReaderWriterLock()
        writer_holding = asyncio.Event()
        reader_acquired = asyncio.Event()

        async def writer():
            await lock.acquire_exclusive()
            writer_holding.set()
            await asyncio.sleep(0.1)
            await lock.release_exclusive()

        async def reader():
            await lock.acquire_shared()
            reader_acquired.set()
            await lock.release_shared()

        writer_task = asyncio.create_task(writer())
        await writer_holding.wait()

        reader_task = asyncio.create_task(reader())
        await asyncio.sleep(0.05)
        assert not reader_acquired.is_set()

        await asyncio.gather(writer_task, reader_task)
        assert reader_acquired.is_set()

    @pytest.mark.asyncio
    async def test_single_writer_exclusive(self):
        lock = ReaderWriterLock()
        writer_count = 3
        completed_writers = 0

        async def writer():
            nonlocal completed_writers
            await lock.acquire_exclusive()
            await asyncio.sleep(0.05)
            completed_writers += 1
            await lock.release_exclusive()

        tasks = [asyncio.create_task(writer()) for _ in range(writer_count)]
        await asyncio.gather(*tasks)

        assert completed_writers == writer_count

    @pytest.mark.asyncio
    async def test_readers_dont_block_each_other(self):
        lock = ReaderWriterLock()
        read_times = []

        async def reader(reader_id):
            start = asyncio.get_event_loop().time()
            await lock.acquire_shared()
            await asyncio.sleep(0.02)
            await lock.release_shared()
            end = asyncio.get_event_loop().time()
            read_times.append((reader_id, end - start))

        tasks = [asyncio.create_task(reader(i)) for i in range(3)]
        await asyncio.gather(*tasks)

        durations = [duration for _, duration in read_times]
        total_duration = sum(durations)

        assert total_duration < 0.15


class TestSchedulerLockAcquisition:
    """Test acquire_locks for each tool effect."""

    @pytest.mark.asyncio
    async def test_read_acquires_shared_on_file_path(self, scheduler):
        """read_file acquires SHARED lock on file:path only (no workspace lock)."""
        call = _make_call(_read_scope("test.py"), {"path": "test.py"})

        async with scheduler.acquire_locks(call) as acquisitions:
            keys = {a.key for a in acquisitions}
            modes = {a.key: a.mode for a in acquisitions}
            assert "file:test.py" in keys
            assert modes["file:test.py"] == LockMode.SHARED
            assert "workspace" not in keys

    @pytest.mark.asyncio
    async def test_write_acquires_exclusive_on_file_path(self, scheduler):
        """replace_exact acquires EXCLUSIVE lock on file:path only (no workspace lock)."""
        call = _make_call(_write_scope("test.py"), {"path": "test.py"})

        async with scheduler.acquire_locks(call) as acquisitions:
            keys = {a.key for a in acquisitions}
            modes = {a.key: a.mode for a in acquisitions}
            assert "file:test.py" in keys
            assert modes["file:test.py"] == LockMode.EXCLUSIVE
            assert "workspace" not in keys

    @pytest.mark.asyncio
    async def test_create_acquires_exclusive_on_file_and_dir(self, scheduler):
        """create_file acquires EXCLUSIVE on file:path and dir:parent (no workspace lock)."""
        call = _make_call(_create_scope("src/new.py"), {"path": "src/new.py"})

        async with scheduler.acquire_locks(call) as acquisitions:
            keys = {a.key for a in acquisitions}
            modes = {a.key: a.mode for a in acquisitions}
            assert "file:src/new.py" in keys
            assert "dir:src" in keys
            assert modes["file:src/new.py"] == LockMode.EXCLUSIVE
            assert modes["dir:src"] == LockMode.EXCLUSIVE
            assert "workspace" not in keys

    @pytest.mark.asyncio
    async def test_delete_acquires_exclusive_on_file_and_dir(self, scheduler):
        """delete_file acquires EXCLUSIVE on file:path and dir:parent (no workspace lock)."""
        call = _make_call(_delete_scope("src/old.py"), {"path": "src/old.py"})

        async with scheduler.acquire_locks(call) as acquisitions:
            keys = {a.key for a in acquisitions}
            modes = {a.key: a.mode for a in acquisitions}
            assert "file:src/old.py" in keys
            assert "dir:src" in keys
            assert modes["file:src/old.py"] == LockMode.EXCLUSIVE
            assert modes["dir:src"] == LockMode.EXCLUSIVE
            assert "workspace" not in keys

    @pytest.mark.asyncio
    async def test_process_acquires_capacity_and_workspace(self, scheduler):
        """run_command acquires processes capacity semaphore and workspace lock per mutation_risk."""
        call = _make_call(_process_scope(), {"mutation_risk": "none"})

        async with scheduler.acquire_locks(call) as acquisitions:
            keys = {a.key for a in acquisitions}
            assert "capacity:processes" in keys
            assert "workspace" in keys
            # Any lock on workspace for local commands is SHARED
            workspace_acq = next(a for a in acquisitions if a.key == "workspace")
            assert workspace_acq.mode == LockMode.SHARED

    @pytest.mark.asyncio
    async def test_process_global_acquires_exclusive_workspace(self, scheduler):
        """run_command with mutation_risk=global acquires EXCLUSIVE workspace lock."""
        call = _make_call(_process_scope(), {"mutation_risk": "global"})

        async with scheduler.acquire_locks(call) as acquisitions:
            keys = {a.key for a in acquisitions}
            assert "capacity:processes" in keys
            assert "workspace" in keys
            workspace_acq = next(a for a in acquisitions if a.key == "workspace")
            assert workspace_acq.mode == LockMode.EXCLUSIVE

    @pytest.mark.asyncio
    async def test_acquires_capacity_semaphore(self, scheduler):
        """acquire_locks acquires capacity semaphore for capacity_group."""
        call = _make_call(_process_scope(), {"mutation_risk": "none"})

        async with scheduler.acquire_locks(call) as acquisitions:
            capacity_acqs = [a for a in acquisitions if a.key.startswith("capacity:")]
            assert len(capacity_acqs) == 1
            assert capacity_acqs[0].key == "capacity:processes"
            assert isinstance(capacity_acqs[0].lock, asyncio.Semaphore)

    @pytest.mark.asyncio
    async def test_sorts_locks_by_key(self, scheduler):
        """acquire_locks sorts resource locks by key for deadlock prevention."""
        # create_file on src/new.py produces dir:src then file:src/new.py,
        # but acquisition order should be sorted by key
        call = _make_call(_create_scope("src/new.py"), {"path": "src/new.py"})

        acquired_order = []

        async def capture_order():
            async with scheduler.acquire_locks(call) as acquisitions:
                acquired_order.extend([a.key for a in acquisitions])

        await capture_order()
        assert acquired_order == sorted(acquired_order)

    @pytest.mark.asyncio
    async def test_releases_locks_on_success(self, scheduler):
        """acquire_locks releases all locks on context exit (success)."""
        call = _make_call(_read_scope("test.py"), {"path": "test.py"})

        async with scheduler.acquire_locks(call):
            pass

        # After release, the same lock should be acquireable immediately
        acquired = asyncio.Event()

        async def reacquire():
            async with scheduler.acquire_locks(call):
                acquired.set()

        await asyncio.wait_for(reacquire(), timeout=1.0)
        assert acquired.is_set()

    @pytest.mark.asyncio
    async def test_releases_locks_on_exception(self, scheduler):
        """acquire_locks releases all locks on context exit (exception)."""
        call = _make_call(_read_scope("test.py"), {"path": "test.py"})

        try:
            async with scheduler.acquire_locks(call):
                raise ValueError("boom")
        except ValueError:
            pass

        # After exception, the lock should be released and acquireable
        acquired = asyncio.Event()

        async def reacquire():
            async with scheduler.acquire_locks(call):
                acquired.set()

        await asyncio.wait_for(reacquire(), timeout=1.0)
        assert acquired.is_set()


class TestSchedulerReaderWriterSemantics:
    """Test reader-writer lock behavior across tool calls."""

    @pytest.mark.asyncio
    async def test_multiple_readers_same_file_concurrent(self, scheduler):
        """Multiple READ tools on same file:path acquire SHARED lock concurrently."""
        call1 = _make_call(_read_scope("test.py"), {"path": "test.py"})
        call2 = _make_call(_read_scope("test.py"), {"path": "test.py"})

        active_readers = 0
        max_concurrent = 0

        async def reader(call):
            nonlocal active_readers, max_concurrent
            async with scheduler.acquire_locks(call):
                active_readers += 1
                max_concurrent = max(max_concurrent, active_readers)
                await asyncio.sleep(0.02)
                active_readers -= 1

        await asyncio.gather(reader(call1), reader(call2))
        assert max_concurrent == 2

    @pytest.mark.asyncio
    async def test_writer_waits_for_readers(self, scheduler):
        """READ tool and WRITE tool on same file:path: writer waits for all readers to release."""
        read_call = _make_call(_read_scope("test.py"), {"path": "test.py"})
        write_call = _make_call(_write_scope("test.py"), {"path": "test.py"})

        reader_holding = asyncio.Event()
        writer_acquired = asyncio.Event()

        async def reader():
            async with scheduler.acquire_locks(read_call):
                reader_holding.set()
                await asyncio.sleep(0.05)

        async def writer():
            await reader_holding.wait()
            async with scheduler.acquire_locks(write_call):
                writer_acquired.set()
                await asyncio.sleep(0.02)

        reader_task = asyncio.create_task(reader())
        await reader_holding.wait()

        writer_task = asyncio.create_task(writer())
        await asyncio.sleep(0.03)
        assert not writer_acquired.is_set()

        await asyncio.gather(reader_task, writer_task)
        assert writer_acquired.is_set()

    @pytest.mark.asyncio
    async def test_multiple_writers_same_file_serialize(self, scheduler):
        """Multiple WRITE tools on same file:path serialize exclusively."""
        call1 = _make_call(_write_scope("test.py"), {"path": "test.py"})
        call2 = _make_call(_write_scope("test.py"), {"path": "test.py"})

        completed = 0

        async def writer(call):
            nonlocal completed
            async with scheduler.acquire_locks(call):
                await asyncio.sleep(0.02)
                completed += 1

        await asyncio.gather(writer(call1), writer(call2))
        assert completed == 2

    @pytest.mark.asyncio
    async def test_readers_different_files_concurrent(self, scheduler):
        """READ tools on different files proceed concurrently."""
        call1 = _make_call(_read_scope("file1.py"), {"path": "file1.py"})
        call2 = _make_call(_read_scope("file2.py"), {"path": "file2.py"})

        start_times = []

        async def reader(call, label):
            start = asyncio.get_event_loop().time()
            async with scheduler.acquire_locks(call):
                await asyncio.sleep(0.02)
            start_times.append((label, asyncio.get_event_loop().time() - start))

        await asyncio.gather(reader(call1, "a"), reader(call2, "b"))
        durations = [d for _, d in start_times]
        total_duration = sum(durations)
        assert total_duration < 0.1

    @pytest.mark.asyncio
    async def test_writers_different_files_concurrent(self, scheduler):
        """WRITE tools on different files proceed concurrently (no shared workspace lock)."""
        call1 = _make_call(_write_scope("file1.py"), {"path": "file1.py"})
        call2 = _make_call(_write_scope("file2.py"), {"path": "file2.py"})

        durations = []

        async def writer(call, label):
            start = asyncio.get_event_loop().time()
            async with scheduler.acquire_locks(call):
                await asyncio.sleep(0.02)
            durations.append((label, asyncio.get_event_loop().time() - start))

        await asyncio.gather(writer(call1, "a"), writer(call2, "b"))
        total_duration = sum(d for _, d in durations)
        assert total_duration < 0.1


class TestSchedulerCapacityGroups:
    """Test capacity semaphore groups."""

    @pytest.mark.asyncio
    async def test_processes_semaphore_limits_concurrency(self, scheduler):
        """processes semaphore (max_parallel_processes=2) limits concurrent run_command."""
        call1 = _make_call(_process_scope(), {"mutation_risk": "none"})
        call2 = _make_call(_process_scope(), {"mutation_risk": "none"})
        call3 = _make_call(_process_scope(), {"mutation_risk": "none"})

        active = 0
        max_concurrent = 0

        async def run(call):
            nonlocal active, max_concurrent
            async with scheduler.acquire_locks(call):
                active += 1
                max_concurrent = max(max_concurrent, active)
                await asyncio.sleep(0.02)
                active -= 1

        await asyncio.gather(run(call1), run(call2), run(call3))
        assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_reject_conflict_rule_raises_when_at_limit(self, scheduler):
        """capacity_group with conflict_rule=REJECT raises ToolConflictError when at limit."""
        # Fill the processes semaphore (capacity 2) with WAIT acquires
        call1 = _make_call(_capacity_scope(), {})
        call2 = _make_call(_capacity_scope(), {})

        async with scheduler.acquire_locks(call1):
            async with scheduler.acquire_locks(call2):
                # Semaphore is now at capacity (2/2)
                call3 = _make_call(
                    ConcurrencyScope(
                        effect=OperationEffect.PROCESS,
                        resource_locks=[],
                        capacity_group="processes",
                        conflict_rule=ConflictRule.REJECT,
                        lifetime=LockLifetime.DETACHED_PROCESS,
                    ),
                    {},
                )
                with pytest.raises(ToolConflictError):
                    async with scheduler.acquire_locks(call3):
                        pass

    @pytest.mark.asyncio
    async def test_wait_conflict_rule_blocks_until_slot_available(self, scheduler):
        """capacity_group with conflict_rule=WAIT blocks until slot available."""
        call1 = _make_call(_capacity_scope(), {})
        call2 = _make_call(_capacity_scope(), {})
        call3 = _make_call(_capacity_scope(), {})

        acquired3 = asyncio.Event()

        async def holder(call):
            async with scheduler.acquire_locks(call):
                await asyncio.sleep(0.05)

        async def waiter(call):
            async with scheduler.acquire_locks(call):
                acquired3.set()

        await asyncio.gather(holder(call1), holder(call2), waiter(call3))
        assert acquired3.is_set()


class TestSchedulerClassification:
    """Test classify_tool_calls grouping."""

    @pytest.mark.asyncio
    async def test_disjoint_resources_same_group(self, scheduler):
        """Calls with disjoint resource keys group together for parallel execution."""
        call1 = _make_call(_read_scope("file1.py"), {"path": "file1.py"})
        call2 = _make_call(_read_scope("file2.py"), {"path": "file2.py"})

        groups = await scheduler.classify_tool_calls([call1, call2])
        assert len(groups) == 1
        assert "group_0" in groups
        assert len(groups["group_0"]) == 2

    @pytest.mark.asyncio
    async def test_overlapping_resources_separate_groups(self, scheduler):
        """Calls with overlapping resource keys go into different groups."""
        call1 = _make_call(_write_scope("test.py"), {"path": "test.py"})
        call2 = _make_call(_write_scope("test.py"), {"path": "test.py"})

        groups = await scheduler.classify_tool_calls([call1, call2])
        assert len(groups) == 2
        assert "group_0" in groups
        assert "group_1" in groups

    @pytest.mark.asyncio
    async def test_preserves_input_order_within_groups(self, scheduler):
        """classify_tool_calls preserves input order within groups."""
        call1 = _make_call(_read_scope("a.py"), {"path": "a.py"})
        call2 = _make_call(_read_scope("b.py"), {"path": "b.py"})
        call3 = _make_call(_read_scope("c.py"), {"path": "c.py"})

        groups = await scheduler.classify_tool_calls([call1, call2, call3])
        group = groups["group_0"]
        assert [g.tool_definition for g in group] == [
            call1.tool_definition,
            call2.tool_definition,
            call3.tool_definition,
        ]

    @pytest.mark.asyncio
    async def test_group_key_format(self, scheduler):
        """Group keys follow 'group_N' format where N is group index."""
        calls = [
            _make_call(_read_scope(f"file{i}.py"), {"path": f"file{i}.py"})
            for i in range(4)
        ]
        # All disjoint -> should all be in group_0
        groups = await scheduler.classify_tool_calls(calls)
        assert list(groups.keys()) == ["group_0"]

    @pytest.mark.asyncio
    async def test_empty_calls_returns_empty_groups(self, scheduler):
        """classify_tool_calls with no calls returns empty dict."""
        groups = await scheduler.classify_tool_calls([])
        assert groups == {}
