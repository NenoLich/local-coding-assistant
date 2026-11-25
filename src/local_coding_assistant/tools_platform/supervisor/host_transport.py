"""Host process transport implementation.

Provides OS-specific subprocess operations for the host backend:
- Subprocess spawning with asyncio.create_subprocess_exec
- Process-tree termination with SIGTERM/SIGKILL (POSIX) or TerminateProcess (Windows)
- Output streaming from stdout/stderr
"""

from __future__ import annotations

import asyncio
import subprocess
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import psutil

from local_coding_assistant.tools_platform.supervisor.transport import (
    ProcessTransport,
    TransportConfig,
)

if TYPE_CHECKING:
    from local_coding_assistant.tools_platform.contracts import (
        ProcessHandle,
    )


class HostProcessTransport(ProcessTransport):
    """Host-specific process transport using asyncio subprocess.

    Handles OS subprocess spawning, termination, and streaming for the host backend.
    """

    def __init__(self) -> None:
        """Initialize the transport with an in-flight process registry."""
        self._processes: dict[int, Any] = {}

    async def spawn(
        self,
        config: TransportConfig,
    ) -> tuple[ProcessHandle, AsyncIterator[bytes], AsyncIterator[bytes]]:
        """Spawn a subprocess and return handle with stdout/stderr iterators.

        Args:
            config: Transport configuration

        Returns:
            Tuple of (process_handle, stdout_iterator, stderr_iterator)

        Raises:
            FileNotFoundError: If executable not found
            Exception: If spawning fails
        """
        from local_coding_assistant.tools_platform.contracts import (
            BackendType,
            ProcessHandle,
            ProcessState,
        )

        # Resolve working directory
        cwd = str(config.cwd) if config.cwd else None

        # Prepare environment
        env = config.env if config.env else None

        # Disable stdin as per requirement
        stdin = subprocess.DEVNULL

        # Create subprocess
        process = await asyncio.create_subprocess_exec(
            *config.argv,
            cwd=cwd,
            env=env,
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Track the process so completion can be awaited and exit codes observed.
        self._processes[process.pid] = process

        # Create process handle
        handle = ProcessHandle(
            handle_id=str(process.pid),  # Use PID as handle ID for host
            backend=BackendType.HOST,
            state=ProcessState.RUNNING,
            os_pid=process.pid,
        )

        # Create stdout iterator
        async def read_stdout() -> AsyncIterator[bytes]:
            """Read from stdout."""
            if process.stdout:
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    yield line

        # Create stderr iterator
        async def read_stderr() -> AsyncIterator[bytes]:
            """Read from stderr."""
            if process.stderr:
                while True:
                    line = await process.stderr.readline()
                    if not line:
                        break
                    yield line

        return handle, read_stdout(), read_stderr()

    async def terminate(
        self,
        handle: ProcessHandle,
        grace: timedelta,
    ) -> None:
        """Terminate a process gracefully then forcefully.

        Args:
            handle: Process handle to terminate
            grace: Grace period before forceful termination

        Raises:
            Exception: If termination fails
        """
        if handle.os_pid is None:
            return

        try:
            # 1. Wrap the system PID in a psutil Process object
            proc = psutil.Process(handle.os_pid)

            # 2. Request graceful termination (SIGTERM on POSIX, TerminateProcess on Windows)
            # Note: Windows doesn't natively have a true SIGTERM, but psutil gracefully
            # triggers standard clean-up handles if the process supports it.
            proc.terminate()

        except psutil.NoSuchProcess:
            return  # The process is already gone

        # 3. Wait out the grace period while checking if it exited
        end_time = asyncio.get_event_loop().time() + grace.total_seconds()
        while asyncio.get_event_loop().time() < end_time:
            if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
                return  # Process exited cleanly!
            await asyncio.sleep(0.1)

        # 4. Grace period expired, force kill it if it's still alive
        try:
            if proc.is_running():
                proc.kill()  # SIGKILL on POSIX, forceful TerminateProcess on Windows
        except psutil.NoSuchProcess:
            pass  # Process managed to close right at the deadline

    async def wait_for_completion(
        self,
        handle: ProcessHandle,
    ) -> int | None:
        """Wait for process completion and return exit code.

        Args:
            handle: Process handle to wait for

        Returns:
            Exit code if process has exited, None if still running

        Raises:
            Exception: If waiting fails
        """
        proc = self._processes.get(handle.os_pid) if handle.os_pid is not None else None
        if proc is None:
            return None
        try:
            # Wait only up to a short interval so the supervision loop stays
            # responsive to hard/idle timeouts for long-running processes.
            return await asyncio.wait_for(proc.wait(), timeout=1.0)
        except TimeoutError:
            return None
        finally:
            # Only drop the tracked process once we have observed its exit code.
            if proc.returncode is not None and handle.os_pid is not None:
                self._processes.pop(handle.os_pid, None)

    def is_running(self, handle: ProcessHandle) -> bool:
        """Check if process is still running.

        Args:
            handle: Process handle to check

        Returns:
            True if process is running, False otherwise
        """
        if handle.os_pid is None:
            return False

        try:
            proc = psutil.Process(handle.os_pid)
            return proc.is_running()
        except psutil.NoSuchProcess:
            return False
        except psutil.AccessDenied:
            return False
