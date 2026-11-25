"""Process transport protocol for backend-specific process operations.

Defines the contract that backends must implement for process management.
The transport layer owns backend-specific operations like subprocess spawning,
process-tree termination, and streaming output handling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from local_coding_assistant.tools_platform.contracts import ProcessHandle


@dataclass
class TransportConfig:
    """Configuration for process transport."""

    argv: list[str]
    cwd: Path | None = None
    env: dict[str, str] | None = None
    input: str | None = None


class ProcessTransport(ABC):
    """Protocol for backend-specific process transport operations.

    Each backend (host, Docker) implements this to provide:
    - Process spawning with backend-specific mechanisms
    - Output streaming from the process
    - Process termination with backend-specific cancellation
    - Process handle management

    The transport does NOT own timeout policy, output limits, or supervision
    logic - those are owned by the shared ExecutionSupervisor.
    """

    @abstractmethod
    async def spawn(
        self, config: TransportConfig
    ) -> tuple[ProcessHandle, AsyncIterator[bytes], AsyncIterator[bytes]]:
        """Spawn a process and return handle with stdout/stderr iterators.

        Args:
            config: Transport configuration

        Returns:
            Tuple of (process_handle, stdout_iterator, stderr_iterator)

        Raises:
            Exception: If spawning fails
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    async def wait_for_completion(self, handle: ProcessHandle) -> int | None:
        """Wait for process completion and return exit code.

        Args:
            handle: Process handle to wait for

        Returns:
            Exit code if process has exited, None if still running

        Raises:
            Exception: If waiting fails
        """
        pass

    @abstractmethod
    def is_running(self, handle: ProcessHandle) -> bool:
        """Check if process is still running.

        Args:
            handle: Process handle to check

        Returns:
            True if process is running, False otherwise
        """
        pass
