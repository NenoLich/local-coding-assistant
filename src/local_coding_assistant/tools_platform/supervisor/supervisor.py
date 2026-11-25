"""Execution supervisor for command execution.

Provides shared execution supervision infrastructure for all backends:
- Concurrent stdout/stderr stream draining to avoid pipe deadlocks
- Soft timeout with progress events and idle timeout detection
- Hard deadline enforcement with configurable grace period
- Output byte/line limits with truncation metadata
- Cancellation support via backend-specific process transport
- Platform-specific return-code mapping (Windows vs POSIX)
- Process handle lifecycle management (spawn, track, cleanup)

The supervisor owns shared policy and lifecycle:
- Deadlines, idle timeouts, output limits
- Stream normalization
- Progress/status events
- Opaque process handles
- Cancellation requests
- State transitions and metrics

Backend-specific operations (subprocess spawning, termination) are delegated
to a ProcessTransport implementation (e.g., HostProcessTransport, DockerProcessTransport).
"""

from __future__ import annotations

import asyncio
import signal
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from local_coding_assistant.tools_platform.contracts import (
    BackendType,
    ProcessHandle,
    ProcessState,
    ToolOutputChunk,
)
from local_coding_assistant.tools_platform.supervisor.transport import (
    ProcessTransport,
    TransportConfig,
)
from local_coding_assistant.utils.logging import get_logger

if TYPE_CHECKING:
    pass


logger = get_logger("tools_platform.supervisor")


@dataclass
class OutputLimits:
    """Output limit configuration."""

    max_stdout_bytes: int = 262144  # 256KB
    max_stderr_bytes: int = 262144  # 256KB
    max_lines: int = 400


@dataclass
class TimeoutPolicy:
    """Timeout policy configuration."""

    soft_timeout: timedelta = timedelta(seconds=10)
    hard_timeout: timedelta = timedelta(seconds=300)
    idle_timeout: timedelta = timedelta(seconds=30)
    heartbeat_interval: timedelta = timedelta(seconds=5)
    cancellation_grace: timedelta = timedelta(seconds=5)


@dataclass
class ExecutionConfig:
    """Configuration for a single execution."""

    argv: list[str]
    cwd: Path | None = None
    env: dict[str, str] | None = None
    input: str | None = None
    timeout_policy: TimeoutPolicy = field(default_factory=TimeoutPolicy)
    output_limits: OutputLimits = field(default_factory=OutputLimits)
    backend: BackendType = BackendType.HOST
    transport: ProcessTransport | None = None  # Backend-specific process transport


class ProcessSupervisor:
    """Supervises process execution with timeout, cancellation, and output limits.

    The supervisor owns shared policy (timeouts, output limits, stream normalization)
    and delegates backend-specific process operations to a ProcessTransport.
    """

    def __init__(self, transport: ProcessTransport | None = None) -> None:
        """Initialize the supervisor.

        Args:
            transport: Backend-specific process transport. If None, a default
                      HostProcessTransport will be used for compatibility.
        """
        if transport is None:
            # Default to host transport for backward compatibility
            from local_coding_assistant.tools_platform.supervisor.host_transport import (
                HostProcessTransport,
            )

            transport = HostProcessTransport()

        self._transport = transport
        self._active_processes: dict[str, ProcessHandle] = {}
        self._process_tasks: dict[str, asyncio.Task[None]] = {}
        self._process_configs: dict[str, ExecutionConfig] = {}

    async def execute(
        self,
        config: ExecutionConfig,
    ) -> AsyncIterator[ToolOutputChunk]:
        """Execute a command with supervision.

        Args:
            config: Execution configuration

        Yields:
            ToolOutputChunk with stdout/stderr/status

        Raises:
            ToolExecutionError: If execution fails
        """
        handle_id = str(uuid.uuid4())
        handle = ProcessHandle(
            handle_id=handle_id,
            backend=config.backend,
            state=ProcessState.SPAWNING,
        )
        self._active_processes[handle_id] = handle
        self._process_configs[handle_id] = config

        try:
            # Use provided transport or config transport
            transport = config.transport or self._transport

            # Spawn the process via transport
            transport_config = TransportConfig(
                argv=config.argv,
                cwd=config.cwd,
                env=config.env,
                input=config.input,
            )
            handle, stdout_iter, stderr_iter = await transport.spawn(transport_config)
            # Expose the generated uuid as the stable external handle id, not the
            # transport's internal os pid, so check_process_status/cancel_execution
            # round-trip against _active_processes (keyed by uuid).
            handle.handle_id = handle_id
            self._active_processes[handle_id] = handle

            # Create supervision task
            supervision_task = asyncio.create_task(
                self._supervise_execution(
                    handle=handle,
                    config=config,
                    transport=transport,
                )
            )
            self._process_tasks[handle_id] = supervision_task

            # Stream output
            async for chunk in self._stream_output(
                stdout_iter=stdout_iter,
                stderr_iter=stderr_iter,
                handle=handle,
                config=config,
            ):
                yield chunk

            # Wait for supervision to complete
            await supervision_task

            # Emit final status chunk
            yield self._create_status_chunk(
                handle=handle,
            )

        except asyncio.CancelledError:
            handle.state = ProcessState.CANCELLED
            # Kill the managed process before propagating so an interrupted
            # tool never leaves an orphaned subprocess behind
            await self._terminate_on_cancellation(handle_id, handle)
            raise
        except FileNotFoundError:
            handle.state = ProcessState.FAILED
            raise
        except Exception:
            handle.state = ProcessState.FAILED
            raise
        finally:
            # Cleanup
            if handle_id in self._active_processes:
                del self._active_processes[handle_id]
            if handle_id in self._process_tasks:
                del self._process_tasks[handle_id]
            self._process_configs.pop(handle_id, None)

    async def cancel(self, handle_id: str) -> ProcessHandle:
        """Cancel a running process.

        Args:
            handle_id: Process handle ID

        Returns:
            Updated process handle

        Raises:
            ValueError: If cancellation fails
        """
        if handle_id not in self._active_processes:
            raise ValueError(f"Process handle not found: {handle_id}")

        handle = self._active_processes[handle_id]

        if handle.state in (
            ProcessState.COMPLETED,
            ProcessState.FAILED,
            ProcessState.CANCELLED,
        ):
            # Already terminated
            return handle

        # Cancel the supervision task
        if handle_id in self._process_tasks:
            self._process_tasks[handle_id].cancel()

        # Terminate the process via transport
        config = self._process_configs.get(handle_id)
        grace = (
            config.timeout_policy.cancellation_grace
            if config is not None
            else TimeoutPolicy().cancellation_grace
        )
        transport = config.transport if config else self._transport
        if transport is not None:
            await transport.terminate(handle, grace=grace)

        handle.state = ProcessState.CANCELLED
        return handle

    async def _terminate_on_cancellation(self, handle_id: str, handle: Any) -> None:
        """Terminate a managed process after interruption.

        Runs on the CancelledError path of execute(): the process is killed
        through its transport (SIGTERM -> cancellation grace -> SIGKILL) before
        the cancellation propagates, so interrupted tools never leave orphaned
        subprocesses. The wait is shielded so a second cancellation cannot skip
        termination.

        Args:
            handle_id: Internal handle ID of the stopped execution
            handle: Process handle to terminate
        """
        config = self._process_configs.get(handle_id)
        transport = (
            config.transport if config is not None else None
        ) or self._transport
        if transport is None:
            return
        grace = (
            config.timeout_policy.cancellation_grace
            if config is not None
            else TimeoutPolicy().cancellation_grace
        )

        terminate = asyncio.ensure_future(transport.terminate(handle, grace=grace))
        try:
            await asyncio.shield(terminate)
        except asyncio.CancelledError:
            # A second cancellation arrived: keep terminating in the background
            pass
        except Exception:
            # Best effort: the process may already be gone
            logger.debug(
                "Process termination after interruption failed",
                handle_id=handle_id,
                exc_info=True,
            )

    async def get_handle(self, handle_id: str) -> ProcessHandle | None:
        """Get a process handle by ID.

        Args:
            handle_id: Process handle ID

        Returns:
            Process handle if found, None otherwise
        """
        return self._active_processes.get(handle_id)

    async def _supervise_execution(
        self,
        handle: ProcessHandle,
        config: ExecutionConfig,
        transport: ProcessTransport,
    ) -> None:
        """Supervise process execution with timeout and cancellation.

        Args:
            handle: Process handle to update
            config: Execution configuration
            transport: Process transport for backend-specific operations
        """
        start_time = datetime.now(UTC)

        try:
            # Wait for process completion with timeout supervision
            while True:
                # Check if process has exited via transport
                return_code = await transport.wait_for_completion(handle)
                if return_code is not None:
                    handle.return_code = return_code
                    if return_code == 0:
                        handle.state = ProcessState.COMPLETED
                    else:
                        handle.state = ProcessState.FAILED
                    break

                # Check if still running
                if not transport.is_running(handle):
                    handle.state = ProcessState.FAILED
                    break

                # Check hard timeout
                elapsed = datetime.now(UTC) - start_time
                if elapsed >= config.timeout_policy.hard_timeout:
                    await transport.terminate(
                        handle,
                        grace=config.timeout_policy.cancellation_grace,
                    )
                    handle.state = ProcessState.TIMED_OUT
                    handle.exit_signal = "SIGKILL" if self._is_posix() else "TERMINATE"
                    break

                # Check idle timeout
                if handle.last_activity:
                    idle_time = datetime.now(UTC) - handle.last_activity
                    if idle_time >= config.timeout_policy.idle_timeout:
                        await transport.terminate(
                            handle,
                            grace=config.timeout_policy.cancellation_grace,
                        )
                        handle.state = ProcessState.TIMED_OUT
                        handle.exit_signal = (
                            "SIGTERM" if self._is_posix() else "TERMINATE"
                        )
                        break

                # Wait for heartbeat interval
                await asyncio.sleep(
                    config.timeout_policy.heartbeat_interval.total_seconds()
                )

        except asyncio.CancelledError:
            # Task was cancelled (likely by cancel() call)
            await transport.terminate(
                handle,
                grace=config.timeout_policy.cancellation_grace,
            )
            handle.state = ProcessState.CANCELLED
            raise

    async def _stream_output(
        self,
        stdout_iter: AsyncIterator[bytes],
        stderr_iter: AsyncIterator[bytes],
        handle: ProcessHandle,
        config: ExecutionConfig,
    ) -> AsyncIterator[ToolOutputChunk]:
        """Stream stdout and stderr concurrently.

        Args:
            stdout_iter: Stdout iterator from transport
            stderr_iter: Stderr iterator from transport
            handle: Process handle to update
            config: Execution configuration

        Yields:
            ToolOutputChunk with output
        """
        stdout_bytes = 0
        stderr_bytes = 0
        stdout_lines = 0
        stderr_lines = 0
        sequence = 0
        cumulative_bytes = 0

        tasks = {
            asyncio.create_task(anext(stdout_iter, None)): "stdout",
            asyncio.create_task(anext(stderr_iter, None)): "stderr",
        }

        while tasks:
            done, _ = await asyncio.wait(
                tasks.keys(),
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done:
                channel = tasks.pop(task)
                data = task.result()

                if data is None:
                    continue

                (
                    stdout_bytes,
                    stderr_bytes,
                    stdout_lines,
                    stderr_lines,
                    truncated,
                ) = self._update_channel_stats(
                    handle,
                    channel,
                    data,
                    stdout_bytes,
                    stderr_bytes,
                    stdout_lines,
                    stderr_lines,
                    config,
                )

                handle.metadata["stdout_bytes"] = stdout_bytes
                handle.metadata["stderr_bytes"] = stderr_bytes
                handle.metadata["stdout_lines"] = stdout_lines
                handle.metadata["stderr_lines"] = stderr_lines

                handle.last_activity = datetime.now(UTC)

                if not truncated:
                    cumulative_bytes += len(data)
                    yield ToolOutputChunk(
                        stream_id=handle.handle_id,
                        sequence=sequence,
                        channel=channel,
                        payload=data.decode("utf-8", errors="replace"),
                        cumulative_bytes=cumulative_bytes,
                    )
                    sequence += 1

                tasks[self._next_read_task(channel, stdout_iter, stderr_iter)] = channel

    def _update_channel_stats(
        self,
        handle: ProcessHandle,
        channel: str,
        data: bytes,
        stdout_bytes: int,
        stderr_bytes: int,
        stdout_lines: int,
        stderr_lines: int,
        config: ExecutionConfig,
    ) -> tuple[int, int, int, int, bool]:
        """Update byte/line counters and truncation metadata for a channel.

        Returns the updated (stdout_bytes, stderr_bytes, stdout_lines, stderr_lines, truncated).
        """
        limits = config.output_limits
        truncated = False

        if channel == "stdout":
            stdout_bytes += len(data)
            stdout_lines += 1
            if stdout_bytes > limits.max_stdout_bytes:
                handle.metadata["stdout_truncated"] = True
            if stdout_lines > limits.max_lines:
                handle.metadata["stdout_truncated"] = True
            truncated = (
                stdout_bytes > limits.max_stdout_bytes
                or stdout_lines > limits.max_lines
            )
        else:
            stderr_bytes += len(data)
            stderr_lines += 1
            if stderr_bytes > limits.max_stderr_bytes:
                handle.metadata["stderr_truncated"] = True
            if stderr_lines > limits.max_lines:
                handle.metadata["stderr_truncated"] = True
            truncated = (
                stderr_bytes > limits.max_stderr_bytes
                or stderr_lines > limits.max_lines
            )

        return stdout_bytes, stderr_bytes, stdout_lines, stderr_lines, truncated

    def _next_read_task(
        self,
        channel: str,
        stdout_iter: AsyncIterator[bytes],
        stderr_iter: AsyncIterator[bytes],
    ) -> asyncio.Task[bytes | None]:
        """Create the next read task for the given channel."""
        if channel == "stdout":
            return asyncio.create_task(anext(stdout_iter, None))
        return asyncio.create_task(anext(stderr_iter, None))

    def _create_status_chunk(
        self,
        handle: ProcessHandle,
    ) -> ToolOutputChunk:
        """Create a status chunk with final process state.

        Args:
            handle: Process handle

        Returns:
            ToolOutputChunk with status
        """
        return ToolOutputChunk(
            stream_id=handle.handle_id,
            sequence=-1,
            channel="status",
            payload={
                "handle_id": handle.handle_id,
                "state": handle.state.value,
                "return_code": handle.return_code,
                "exit_signal": handle.exit_signal,
                "os_pid": handle.os_pid,
                "truncated": bool(
                    handle.metadata.get("stdout_truncated")
                    or handle.metadata.get("stderr_truncated")
                ),
                "stdout_truncated": bool(handle.metadata.get("stdout_truncated")),
                "stderr_truncated": bool(handle.metadata.get("stderr_truncated")),
                "stdout_bytes": handle.metadata.get("stdout_bytes", 0),
                "stderr_bytes": handle.metadata.get("stderr_bytes", 0),
                "stdout_lines": handle.metadata.get("stdout_lines", 0),
                "stderr_lines": handle.metadata.get("stderr_lines", 0),
            },
        )

    def _is_posix(self) -> bool:
        """Check if running on POSIX system."""
        import sys

        return sys.platform != "win32"

    def map_return_code(self, return_code: int) -> dict[str, Any]:
        """Map return code to platform-specific interpretation.

        Args:
            return_code: Process return code

        Returns:
            Dictionary with code interpretation
        """
        if self._is_posix():
            return self._map_posix_return_code(return_code)
        else:
            return self._map_windows_return_code(return_code)

    def _map_posix_return_code(self, return_code: int) -> dict[str, Any]:
        """Map POSIX return code.

        Args:
            return_code: Process return code

        Returns:
            Dictionary with code interpretation
        """
        if return_code == 0:
            return {"success": True, "signal": None}
        elif return_code < 0:
            # Terminated by signal
            signal_name = signal.Signals(-return_code).name
            return {"success": False, "signal": signal_name}
        else:
            # Exited with code
            return {"success": False, "signal": None, "exit_code": return_code}

    def _map_windows_return_code(self, return_code: int) -> dict[str, Any]:
        """Map Windows return code.

        Args:
            return_code: Process return code

        Returns:
            Dictionary with code interpretation
        """
        if return_code == 0:
            return {"success": True, "signal": None}
        else:
            # Windows exit codes are less structured
            return {"success": False, "signal": None, "exit_code": return_code}
