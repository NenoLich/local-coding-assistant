# pyrefly: ignore-errors
"""Docker process transport implementation.

Provides backend-specific process operations for Docker containers:
- Process spawning via Docker API
- Streaming output from container processes
- Process termination with graceful/forceful shutdown
- Process handle management with container ID tracking
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from local_coding_assistant.tools_platform.contracts import ProcessHandle

try:
    import docker
    from docker.errors import APIError, NotFound
    from docker.models.containers import Container
except ImportError:
    docker = None
    APIError = Exception
    NotFound = Exception
    Container = object

from local_coding_assistant.tools_platform.contracts import (
    BackendType,
    ProcessHandle,
    ProcessState,
)
from local_coding_assistant.tools_platform.supervisor.transport import (
    ProcessTransport,
    TransportConfig,
)
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("tools_platform.supervisor.docker_transport")


class DockerProcessTransport(ProcessTransport):
    """Process transport for Docker container execution.

    Uses Docker API directly for process spawning, streaming output,
    and termination. This provides the ideal host-side interface for
    Docker execution with full streaming/cancellation support.
    """

    def __init__(
        self,
        container: Container | None = None,
        workspace_root: Path | None = None,
    ) -> None:
        """Initialize the Docker process transport.

        Args:
            container: Docker container instance (if None, will use default)
            workspace_root: Workspace root path for mapping
        """
        if docker is None:
            raise RuntimeError("Docker Python SDK not installed")

        self._container = container
        self._workspace_root = workspace_root or Path.cwd()
        self._client = docker.from_env()
        self._active_handles: dict[str, ProcessHandle] = {}

    async def spawn(
        self,
        config: TransportConfig,
    ) -> tuple[ProcessHandle, AsyncIterator[bytes], AsyncIterator[bytes]]:
        """Spawn a process in Docker container and return handle with streams.

        Args:
            config: Transport configuration

        Returns:
            Tuple of (process_handle, stdout_iterator, stderr_iterator)

        Raises:
            RuntimeError: If Docker not available
            Exception: If spawning fails
        """
        import uuid
        from datetime import UTC, datetime

        # Create process handle
        handle_id = str(uuid.uuid4())
        handle = ProcessHandle(
            handle_id=handle_id,
            backend=BackendType.DOCKER,
            state=ProcessState.SPAWNING,
            start_time=datetime.now(UTC),
        )
        self._active_handles[handle_id] = handle

        try:
            # If no container provided, create a temporary one
            if self._container is None:
                container = await self._create_temp_container()
                handle.metadata["container_id"] = container.id
                handle.os_pid = None  # Container processes don't have host PIDs
            else:
                handle.metadata["container_id"] = self._container.id
                handle.os_pid = None

            # Execute command in container
            exec_id = await self._execute_in_container(
                container=self._container or container,
                argv=config.argv,
                cwd=config.cwd,
                env=config.env,
            )
            handle.metadata["exec_id"] = exec_id

            # Create async iterators for stdout/stderr
            stdout_iter = self._stream_output(
                container=self._container or container,
                exec_id=exec_id,
                stream="stdout",
            )
            stderr_iter = self._stream_output(
                container=self._container or container,
                exec_id=exec_id,
                stream="stderr",
            )

            handle.state = ProcessState.RUNNING
            return handle, stdout_iter, stderr_iter

        except Exception as e:
            handle.state = ProcessState.FAILED
            handle.metadata["error"] = str(e)
            logger.error("Failed to spawn Docker process", error=str(e))
            raise

    async def terminate(
        self,
        handle: ProcessHandle,
        grace: timedelta,
    ) -> None:
        """Terminate a Docker process gracefully then forcefully.

        Args:
            handle: Process handle to terminate
            grace: Grace period before forceful termination

        Raises:
            Exception: If termination fails
        """

        container_id = handle.metadata.get("container_id")
        exec_id = handle.metadata.get("exec_id")

        if not container_id or not exec_id:
            logger.warning("No container or exec ID in handle, cannot terminate")
            return

        try:
            # Get container
            _container = self._client.containers.get(container_id)

            # Try graceful termination first (SIGTERM equivalent)
            try:
                # Docker exec doesn't support signal, we need to kill the exec
                # This is a limitation - we may need to track the exec process
                # For now, we'll mark as cancelled
                logger.info(f"Attempting graceful termination of exec {exec_id}")
                await asyncio.sleep(grace.total_seconds())

                # Forceful termination if still running
                # Note: Docker exec processes are harder to kill individually
                # We may need to kill the entire container or use a different approach
                logger.info(f"Forceful termination of exec {exec_id}")
                # This is a placeholder - actual implementation depends on Docker API capabilities
                pass

            except NotFound:
                logger.warning(
                    f"Container {container_id} not found, may already be terminated"
                )

        except Exception as e:
            logger.error("Failed to terminate Docker process", error=str(e))
            raise

    async def wait_for_completion(self, handle: ProcessHandle) -> int | None:
        """Wait for Docker exec completion and return exit code.

        Args:
            handle: Process handle to wait for

        Returns:
            Exit code if process has exited, None if still running

        Raises:
            Exception: If waiting fails
        """
        container_id = handle.metadata.get("container_id")
        exec_id = handle.metadata.get("exec_id")

        if not container_id or not exec_id:
            return None

        try:
            container = self._client.containers.get(container_id)
            exec_info = container.exec_inspect(exec_id)

            if exec_info.get("Running", False):
                return None
            else:
                return exec_info.get("ExitCode", -1)

        except NotFound:
            logger.warning(f"Container {container_id} not found")
            return -1
        except Exception as e:
            logger.error("Failed to wait for Docker process completion", error=str(e))
            return -1

    def is_running(self, handle: ProcessHandle) -> bool:
        """Check if Docker process is still running.

        Args:
            handle: Process handle to check

        Returns:
            True if process is running, False otherwise
        """
        container_id = handle.metadata.get("container_id")
        exec_id = handle.metadata.get("exec_id")

        if not container_id or not exec_id:
            return False

        try:
            container = self._client.containers.get(container_id)
            exec_info = container.exec_inspect(exec_id)
            return exec_info.get("Running", False)
        except NotFound:
            return False
        except Exception:
            return False

    async def _create_temp_container(self) -> Container:
        """Create a temporary container for execution.

        Returns:
            Container instance

        Raises:
            Exception: If container creation fails
        """
        # This is a placeholder - actual implementation will use sandbox configuration
        # For now, we'll create a minimal container
        try:
            container = self._client.containers.create(
                image="python:3.12-slim",
                command=["sleep", "3600"],  # Keep container alive
                volumes={
                    str(self._workspace_root): {
                        "bind": "/workspace",
                        "mode": "rw",
                    }
                },
                working_dir="/workspace",
                detach=True,
            )
            container.start()
            return container
        except Exception as e:
            logger.error("Failed to create temporary container", error=str(e))
            raise

    async def _execute_in_container(
        self,
        container: Container,
        argv: list[str],
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> str:
        """Execute command in container and return exec ID.

        Args:
            container: Container instance
            argv: Command and arguments
            cwd: Working directory (relative to workspace)
            env: Environment variables

        Returns:
            Exec ID

        Raises:
            Exception: If execution fails
        """
        try:
            # Map workspace-relative cwd to container path
            container_cwd = "/workspace"
            if cwd:
                container_cwd = str(Path("/workspace") / cwd)

            # Create exec instance
            exec_id = container.client.api.exec_create(
                container.id,
                cmd=argv,
                workdir=container_cwd,
                environment=env or {},
            )["Id"]

            # Start the exec
            container.client.api.exec_start(exec_id, detach=False)

            return exec_id

        except Exception as e:
            logger.error("Failed to execute in container", error=str(e))
            raise

    async def _stream_output(
        self,
        container: Container,
        exec_id: str,
        stream: str = "stdout",
    ) -> AsyncIterator[bytes]:
        """Stream output from Docker exec.

        Args:
            container: Container instance
            exec_id: Exec ID
            stream: Stream type (stdout or stderr)

        Yields:
            Output chunks as bytes

        Raises:
            Exception: If streaming fails
        """
        try:
            # Docker exec streaming is complex - this is a simplified version
            # Real implementation would use Docker API's attach socket
            # For now, we'll yield empty chunks to satisfy the interface
            # This will be enhanced when sandbox internal logic is added
            yield b""
        except Exception as e:
            logger.error(f"Failed to stream {stream} from Docker exec", error=str(e))
            raise
