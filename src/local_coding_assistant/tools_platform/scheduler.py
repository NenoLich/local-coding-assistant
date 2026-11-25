"""Scheduler for effect classification and async lock management.

This module implements the scheduling middleware that:
- Classifies tool calls by their concurrency scope
- Acquires resource locks based on operation effects
- Manages capacity groups with semaphores
- Handles conflict resolution rules
- Ensures proper lock ordering to prevent deadlocks
- Supports shared reader locks for concurrent read operations
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from local_coding_assistant.tools_platform.contracts import (
    ConcurrencyScope,
    ConflictRule,
    LockLifetime,
    LockMode,
    OperationEffect,
    ResourceLock,
    ValidatedToolCall,
)
from local_coding_assistant.tools_platform.exceptions import ToolConflictError
from local_coding_assistant.tools_platform.workspace.manager import WorkspaceManager
from local_coding_assistant.utils.logging import get_logger

if TYPE_CHECKING:
    from local_coding_assistant.core.protocols import IConfigManager

logger = get_logger("tools_platform.scheduler")


class ReaderWriterLock:
    """Async reader-writer lock supporting multiple concurrent readers.

    Multiple readers can hold the lock simultaneously, but writers require
    exclusive access and must wait for all readers to release.

    This implementation uses:
    - A condition variable for coordination between readers and writers
    - A counter for active readers
    - A flag for active writer
    """

    def __init__(self) -> None:
        """Initialize the reader-writer lock."""
        self._readers = 0
        self._writer_active = False
        self._condition = asyncio.Condition()

    async def acquire_shared(self) -> None:
        """Acquire the lock in shared (read) mode.

        Multiple readers can hold this simultaneously. Readers wait
        if a writer is currently active.
        """
        async with self._condition:
            # Wait if a writer is active
            while self._writer_active:
                await self._condition.wait()
            self._readers += 1

    async def release_shared(self) -> None:
        """Release the lock from shared (read) mode."""
        async with self._condition:
            self._readers -= 1
            if self._readers == 0:
                # Notify waiting writers
                self._condition.notify_all()

    async def acquire_exclusive(self) -> None:
        """Acquire the lock in exclusive (write) mode.

        Writers wait for all active readers to release before proceeding.
        Only one writer can hold the lock at a time.
        """
        async with self._condition:
            # Wait for all readers and any active writer to finish
            while self._writer_active or self._readers > 0:
                await self._condition.wait()
            self._writer_active = True

    async def release_exclusive(self) -> None:
        """Release the lock from exclusive (write) mode."""
        async with self._condition:
            self._writer_active = False
            # Notify waiting readers and writers
            self._condition.notify_all()


@dataclass
class LockAcquisition:
    """Represents an acquired lock that will be released.

    Attributes:
        key: Resource key that was locked
        mode: Lock mode that was acquired
        lock: The underlying asyncio lock or ReaderWriterLock
        lifetime: When this lock should be released
    """

    key: str
    mode: LockMode
    lock: asyncio.Lock | asyncio.Semaphore | ReaderWriterLock
    lifetime: LockLifetime


class Scheduler:
    """Manages concurrent tool execution with effect-based locking.

    The scheduler classifies tool calls by their concurrency scope and
    acquires appropriate locks to ensure safe concurrent execution.
    """

    def __init__(
        self,
        workspace_manager: WorkspaceManager,
        config_manager: IConfigManager,
    ):
        """Initialize the scheduler.

        Args:
            workspace_manager: Workspace manager for path-based operations
            config_manager: Config manager for dynamic configuration resolution
        """
        self._workspace_manager = workspace_manager
        self._config_manager = config_manager

        # Resource locks (key -> ReaderWriterLock for shared reader semantics)
        self._resource_locks: dict[str, ReaderWriterLock] = {}

        # Capacity semaphores with dynamic config
        self._capacity_groups: dict[str, asyncio.Semaphore] = {
            "repository_reads": asyncio.Semaphore(self._get_max_parallel_reads()),
            "processes": asyncio.Semaphore(self._get_max_parallel_processes()),
        }

        # Lock for managing the resource_locks dict itself
        self._registry_lock = asyncio.Lock()

    def _get_max_parallel_reads(self) -> int:
        """Get max parallel reads from config."""
        if self._config_manager is None:
            return 8
        return self._config_manager.global_config.tools_platform.concurrency.max_parallel_reads

    def _get_max_parallel_processes(self) -> int:
        """Get max parallel processes from config."""
        if self._config_manager is None:
            return 2
        return self._config_manager.global_config.tools_platform.concurrency.max_parallel_processes

    @asynccontextmanager
    async def acquire_locks(
        self, validated_call: ValidatedToolCall
    ) -> AsyncGenerator[list[LockAcquisition]]:
        """Acquire locks for a tool call based on its concurrency scope.

        This context manager acquires all required locks in a stable order
        to prevent deadlocks, and releases them on exit.

        Args:
            validated_call: The validated tool call to acquire locks for

        Yields:
            List of acquired locks

        Raises:
            ToolError: If lock acquisition fails due to conflict rule REJECT
        """
        tool_definition = validated_call.tool_definition
        concurrency_scope = tool_definition.concurrency_scope

        if concurrency_scope is None:
            # Default conservative behavior: exclusive workspace lock
            concurrency_scope = ConcurrencyScope(
                effect=OperationEffect.WRITE,
                resource_locks=[ResourceLock(key="workspace", mode=LockMode.EXCLUSIVE)],
                conflict_rule=ConflictRule.WAIT,
                lifetime=LockLifetime.EXECUTION,
            )

        # Compute the actual locks needed
        locks_needed = self._compute_locks(concurrency_scope, validated_call)

        # Acquire capacity semaphore if specified
        capacity_lock: LockAcquisition | None = None
        if concurrency_scope.capacity_group:
            capacity_lock = await self._acquire_capacity(
                concurrency_scope.capacity_group, concurrency_scope.conflict_rule
            )

        # Acquire resource locks in stable order
        acquired_locks: list[LockAcquisition] = []
        try:
            if capacity_lock:
                acquired_locks.append(capacity_lock)

            # Every caller acquires resource locks in the same order. This is
            # required when parallel calls have overlapping resource sets.
            # Sorting by key ensures consistent acquisition order to prevent deadlocks.
            # For reader-writer locks on the same key, the ReaderWriterLock class
            # handles coordination between readers and writers internally.
            for resource_lock in sorted(locks_needed, key=lambda lock: lock.key):
                acquired = await self._acquire_resource_lock(
                    resource_lock, concurrency_scope.conflict_rule
                )
                acquired_locks.append(acquired)

            yield acquired_locks

        finally:
            # Release locks in reverse order
            for acquired in reversed(acquired_locks):
                await self._release_lock(acquired)

    def _compute_locks(
        self, concurrency_scope: ConcurrencyScope, validated_call: ValidatedToolCall
    ) -> list[ResourceLock]:
        """Compute the actual resource locks needed for a call.

        Args:
            concurrency_scope: The tool's concurrency scope
            validated_call: The validated tool call

        Returns:
            List of ResourceLock specifications
        """
        resource_locks = concurrency_scope.resource_locks

        # If resource_locks is a callable, compute from arguments
        if callable(resource_locks):
            lock_factory = cast(
                Callable[[dict[str, object]], list[ResourceLock]], resource_locks
            )
            arguments = validated_call.input_model.model_dump()
            return lock_factory(arguments)

        return cast(list[ResourceLock], resource_locks)

    async def _acquire_capacity(
        self, group: str, conflict_rule: ConflictRule
    ) -> LockAcquisition:
        """Acquire a capacity semaphore.

        Args:
            group: Capacity group name
            conflict_rule: How to handle conflicts

        Returns:
            LockAcquisition for the capacity semaphore

        Raises:
            ToolError: If conflict rule is REJECT and semaphore is full
        """
        async with self._registry_lock:
            if group not in self._capacity_groups:
                # Create semaphore with default capacity if not exists
                self._capacity_groups[group] = asyncio.Semaphore(8)

        semaphore = self._capacity_groups[group]

        if conflict_rule == ConflictRule.REJECT:
            if semaphore.locked():
                raise ToolConflictError(
                    message=f"Capacity group '{group}' is at limit",
                    details={"group": group},
                )
            await semaphore.acquire()
        else:
            # Blocking acquire (WAIT)
            await semaphore.acquire()

        return LockAcquisition(
            key=f"capacity:{group}",
            mode=LockMode.SHARED,
            lock=semaphore,
            lifetime=LockLifetime.EXECUTION,
        )

    async def _acquire_resource_lock(
        self, resource_lock: ResourceLock, conflict_rule: ConflictRule
    ) -> LockAcquisition:
        """Acquire a single resource lock.

        Args:
            resource_lock: The resource lock specification
            conflict_rule: How to handle conflicts

        Returns:
            LockAcquisition for the resource

        Raises:
            ToolError: If conflict rule is REJECT and lock is held
        """
        key = resource_lock.key
        mode = resource_lock.mode

        if mode == LockMode.NONE:
            # No lock needed
            return LockAcquisition(
                key=key,
                mode=mode,
                lock=asyncio.Lock(),  # Dummy lock
                lifetime=LockLifetime.EXECUTION,
            )

        # Handle SHARED and EXCLUSIVE modes with reader-writer semantics
        # Get or create the ReaderWriterLock for this resource
        async with self._registry_lock:
            if key not in self._resource_locks:
                self._resource_locks[key] = ReaderWriterLock()

        lock = self._resource_locks[key]

        if mode == LockMode.SHARED:
            # Acquire shared read lock - multiple readers can hold simultaneously
            if conflict_rule == ConflictRule.REJECT:
                # Check if writer is active without blocking
                async with lock._condition:
                    if lock._writer_active:
                        raise ToolConflictError(
                            message=f"Resource '{key}' has exclusive lock held",
                            details={"key": key, "mode": mode.value},
                        )
                    lock._readers += 1
            else:
                # Blocking acquire (WAIT)
                await lock.acquire_shared()

        elif mode == LockMode.EXCLUSIVE:
            # Acquire exclusive write lock - waits for all readers to release
            if conflict_rule == ConflictRule.REJECT:
                # Check if any readers or writer is active without blocking
                async with lock._condition:
                    if lock._readers > 0 or lock._writer_active:
                        raise ToolConflictError(
                            message=f"Resource '{key}' is locked",
                            details={"key": key, "mode": mode.value},
                        )
                    lock._writer_active = True
            else:
                # Blocking acquire (WAIT)
                await lock.acquire_exclusive()

        return LockAcquisition(
            key=key,
            mode=mode,
            lock=lock,
            lifetime=LockLifetime.EXECUTION,
        )

    async def _release_lock(self, acquisition: LockAcquisition) -> None:
        """Release an acquired lock.

        Args:
            acquisition: The lock acquisition to release
        """
        if acquisition.mode == LockMode.NONE:
            return

        if acquisition.key.startswith("capacity:"):
            # Release semaphore
            cast(asyncio.Semaphore, acquisition.lock).release()
        elif isinstance(acquisition.lock, ReaderWriterLock):
            # Release reader-writer lock based on mode
            if acquisition.mode == LockMode.SHARED:
                await acquisition.lock.release_shared()
            elif acquisition.mode == LockMode.EXCLUSIVE:
                await acquisition.lock.release_exclusive()
        else:
            # Release regular asyncio.Lock
            cast(asyncio.Lock, acquisition.lock).release()

    async def classify_tool_calls(
        self, validated_calls: list[ValidatedToolCall]
    ) -> dict[str, list[ValidatedToolCall]]:
        """Classify tool calls into independent groups for concurrent execution.

        Groups calls by their resource requirements. Calls with disjoint
        resource keys can run concurrently.

        Args:
            validated_calls: List of validated tool calls

        Returns:
            Dictionary mapping group names to lists of tool calls
        """
        groups: dict[str, list[ValidatedToolCall]] = {}
        used_resources: dict[str, list[str]] = {}  # group -> list of resource keys

        for call in validated_calls:
            # Compute resource keys for this call
            resource_keys = self._get_resource_keys(call)

            # Find an existing group with disjoint resources
            assigned_group = None
            for group_name, group_resources in used_resources.items():
                if not self._resources_conflict(resource_keys, group_resources):
                    assigned_group = group_name
                    # Merge resources
                    used_resources[group_name].extend(resource_keys)
                    break

            if assigned_group is None:
                # Create new group
                group_name = f"group_{len(groups)}"
                groups[group_name] = [call]
                used_resources[group_name] = resource_keys
            else:
                groups[assigned_group].append(call)

        logger.debug(
            "Classified %d tool calls into %d groups",
            len(validated_calls),
            len(groups),
        )

        return groups

    def _get_resource_keys(self, validated_call: ValidatedToolCall) -> list[str]:
        """Extract resource keys from a tool call.

        Args:
            validated_call: The validated tool call

        Returns:
            List of resource keys
        """
        tool_definition = validated_call.tool_definition
        concurrency_scope = tool_definition.concurrency_scope

        if concurrency_scope is None:
            return ["workspace"]

        resource_locks = concurrency_scope.resource_locks

        if callable(resource_locks):
            lock_factory = cast(
                Callable[[dict[str, object]], list[ResourceLock]], resource_locks
            )
            arguments = validated_call.input_model.model_dump()
            resource_locks = lock_factory(arguments)

        return [
            key
            for key in (lock.key for lock in cast(list[ResourceLock], resource_locks))
            if key != "workspace"
            and not key.startswith("capacity:")
            and key != "processes"
        ]

    def _resources_conflict(self, keys1: list[str], keys2: list[str]) -> bool:
        """Check if two sets of resource keys conflict.

        Args:
            keys1: First set of resource keys
            keys2: Second set of resource keys

        Returns:
            True if resources conflict, False otherwise
        """
        # Check for any overlapping keys
        return bool(set(keys1) & set(keys2))
