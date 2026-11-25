"""Execution backend protocol interface.

Defines the contract that all execution backends must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from local_coding_assistant.tools_platform.contracts import (
    FileObservation,
    ToolOutputChunk,
)


class ExecutionBackend(ABC):
    """Protocol for execution backends (host, Docker, remote)."""

    @abstractmethod
    async def read_file(
        self,
        path: Path,
        start_line: int | None = None,
        end_line: int | None = None,
        byte_limit: int | None = None,
    ) -> str:
        """Read file content with optional line range and byte limit.

        Args:
            path: Path to file (relative to workspace root)
            start_line: Optional start line (1-indexed)
            end_line: Optional end line (1-indexed)
            byte_limit: Maximum bytes to read

        Returns:
            File content

        Raises:
            ToolExecutionError: If read fails
        """
        pass

    @abstractmethod
    async def observe_file(
        self, path: Path, session_id: str | None = None
    ) -> FileObservation:
        """Observe file state for conflict detection.

        Args:
            path: Path to file (relative to workspace root)
            session_id: Optional session ID to scope the file-version observation.

        Returns:
            FileObservation with current file state

        Raises:
            ToolExecutionError: If observation fails
        """
        pass

    @abstractmethod
    async def create_file(
        self,
        path: str,
        content: str,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Implementation of create_file tool.

        Args:
            path: Path to file relative to workspace root
            content: File content to write
            overwrite: Overwrite the file if it already exists

        Returns:
            Dict with change details
        """
        pass

    @abstractmethod
    async def delete_file(
        self,
        path: str,
    ) -> dict[str, Any]:
        """Implementation of delete_file tool.

        Args:
            path: Path to file relative to workspace root

        Returns:
            Dict with change details
        """
        pass

    @abstractmethod
    def execute_command(
        self,
        argv: list[str],
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[ToolOutputChunk]:
        """Execute a command with streaming output.

        Args:
            argv: Command and arguments
            cwd: Working directory (relative to workspace root)
            env: Environment variables
            timeout_seconds: Timeout in seconds

        Yields:
            ToolOutputChunk with stdout/stderr/status

        Raises:
            ToolExecutionError: If execution fails
        """
        pass

    @abstractmethod
    async def cancel_execution(self, execution_id: str) -> None:
        """Cancel a running execution.

        Args:
            execution_id: Execution identifier

        Raises:
            ToolExecutionError: If cancellation fails
        """
        pass

    @abstractmethod
    async def check_process_status(self, handle_id: str) -> dict[str, Any]:
        """Check status of a LOCCA-owned process by handle.

        Args:
            handle_id: Process handle ID

        Returns:
            Dict with process status information

        Raises:
            ToolExecutionError: If handle not found or check fails
        """
        pass

    @abstractmethod
    async def query_repository(
        self,
        query_type: str,
        params: dict[str, Any],
    ) -> Any:
        """Query repository index (symbols, etc.).

        Args:
            query_type: Type of query (e.g., "search_symbols")
            params: Query parameters

        Returns:
            Query results

        Raises:
            ToolExecutionError: If query fails or index unavailable
        """
        pass

    @abstractmethod
    async def search_text(
        self,
        query: str,
        path: Path,
        pattern: str = "literal",
        includes: list[str] | None = None,
        excludes: list[str] | None = None,
        case_sensitive: bool = False,
        context_lines: int = 2,
        max_matches: int = 200,
        max_bytes: int = 262144,
    ) -> Any:
        """Search for text in files.

        Args:
            query: Search query (literal or regex)
            path: Search path relative to workspace root
            pattern: Pattern type (literal or regex)
            includes: Glob patterns to include
            excludes: Glob patterns to exclude
            case_sensitive: Case-sensitive search
            context_lines: Number of context lines before/after match
            max_matches: Maximum number of matches to return
            max_bytes: Maximum output bytes

        Returns:
            Search results

        Raises:
            ToolExecutionError: If search fails
        """
        pass

    @abstractmethod
    async def match_paths(
        self,
        pattern: str,
        path: Path,
        include_hidden: bool = False,
        include_ignored: bool = False,
        file_type: str | None = None,
        max_results: int = 200,
    ) -> Any:
        """Match files using glob patterns.

        Args:
            pattern: Glob pattern to match files
            path: Search path relative to workspace root
            include_hidden: Include hidden files/directories
            include_ignored: Include git-ignored files
            file_type: Filter by file extension
            max_results: Maximum number of results to return

        Returns:
            Matched paths

        Raises:
            ToolExecutionError: If matching fails
        """
        pass

    @abstractmethod
    async def replace_exact(
        self,
        path: str,
        old_text: str,
        new_text: str,
        session_id: str,
        expected_match_count: int = 1,
        turn_id: str | None = None,
    ) -> dict[str, Any]:
        """Implementation of replace_exact tool.

        Args:
            path: Path to file relative to workspace root
            old_text: Exact text to replace (must match exactly)
            new_text: New text to replace with
            expected_match_count: Expected number of old_text matches
            session_id: Session ID for tracking
            turn_id: Turn ID for tracking

        Returns:
            Dict with change details
        """
        pass

    @abstractmethod
    def get_workspace_root(self) -> Path:
        """Get the workspace root path.

        Returns:
            Path to workspace root
        """
        pass

    @abstractmethod
    def resolve_path(self, path: Path) -> Path:
        """Resolve a path relative to workspace root.

        Args:
            path: Path to resolve (can be relative or absolute)

        Returns:
            Absolute path within workspace

        Raises:
            ToolExecutionError: If path is outside workspace
        """
        pass

    @abstractmethod
    def get_backend_type(self) -> str:
        """Get the backend type identifier."""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the backend (e.g., start sandbox)."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the backend (e.g., stop sandbox)."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available."""
        pass
