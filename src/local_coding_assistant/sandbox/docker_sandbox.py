"""Docker-based sandbox implementation."""

import asyncio
import json
import time
import traceback
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # These imports are only for type checking and not executed at runtime
    import docker
    from docker.models.containers import Container

    from local_coding_assistant.config.schemas import SandboxConfig
else:
    # At runtime, try to import docker, but allow it to fail gracefully
    try:
        import docker
        from docker.errors import APIError, DockerException, ImageNotFound, NotFound
        from docker.models.containers import Container
    except ImportError:
        docker = None
        DockerException = Exception
        NotFound = Exception
        Container = Any
        APIError = Exception
        ImageNotFound = Exception


from local_coding_assistant.config.path_manager import PathManager
from local_coding_assistant.utils.logging import get_logger

from .base import ISandbox
from .exceptions import (
    SandboxOutputFormatError,
    SandboxRuntimeError,
    SandboxSecurityError,
    SandboxTimeoutError,
)
from .sandbox_types import (
    ResourceMetric,
    ResourceType,
    SandboxExecutionRequest,
    SandboxExecutionResponse,
    ToolCallMetric,
)
from .security import SecurityManager

logger = get_logger("sandbox.docker")


@dataclass(frozen=True)
class SandboxPaths:
    """Resolved directories required by the sandbox."""

    project_root: Path
    host_workspace: Path
    tools_src_dir: Path
    workspace_tools_dir: Path
    logs_dir: Path


@dataclass(frozen=True)
class SandboxSettings:
    """Immutable runtime settings for the sandbox."""

    image: str
    timeout: int
    mem_limit: str
    cpu_quota: int
    network_enabled: bool
    auto_build: bool
    max_sessions: int


@dataclass()
class SandboxRuntimeState:
    """Mutable state for DockerSandbox runtime."""

    client: "docker.DockerClient | None" = None
    containers: dict[str, Container] = field(default_factory=dict)
    initialized: bool = False
    paths: SandboxPaths | None = None


class DockerSandbox(ISandbox):
    """Docker-based sandbox implementation."""

    def __init__(
        self,
        image: str = "locca-sandbox:latest",
        timeout: int = 300,  # Default timeout in seconds (5 minutes)
        mem_limit: str = "512m",
        cpu_quota: int = 50000,  # 50% of one CPU
        network_enabled: bool = False,
        security_manager: SecurityManager | None = None,
        auto_build: bool = True,
        path_manager: PathManager | None = None,
        max_sessions: int = 5,
        config: "SandboxConfig | None" = None,
    ):
        self._settings = SandboxSettings(
            image=image,
            timeout=timeout,
            mem_limit=mem_limit,
            cpu_quota=cpu_quota,
            network_enabled=network_enabled,
            auto_build=auto_build,
            max_sessions=max_sessions,
        )
        self.security_manager = security_manager or SecurityManager()
        self._path_manager = path_manager or PathManager()
        self.config = config  # Store the config for downstream consumers
        self._state = SandboxRuntimeState()

    @property
    def image(self) -> str:
        return self._settings.image

    @property
    def timeout(self) -> int:
        return self._settings.timeout

    @property
    def mem_limit(self) -> str:
        return self._settings.mem_limit

    @property
    def cpu_quota(self) -> int:
        return self._settings.cpu_quota

    @property
    def network_enabled(self) -> bool:
        return self._settings.network_enabled

    @property
    def auto_build(self) -> bool:
        return self._settings.auto_build

    @property
    def max_sessions(self) -> int:
        return self._settings.max_sessions

    @property
    def _client(self):
        return self._state.client

    @_client.setter
    def _client(self, client):
        self._state.client = client

    @property
    def _containers(self) -> dict[str, Container]:
        return self._state.containers

    @property
    def _initialized(self) -> bool:
        return self._state.initialized

    @_initialized.setter
    def _initialized(self, value: bool) -> None:
        self._state.initialized = value

    def _paths_or_raise(self) -> SandboxPaths:
        if not self._state.paths:
            raise SandboxRuntimeError(
                "Sandbox paths have not been initialized. Call ensure_directories first."
            )
        return self._state.paths

    @property
    def _project_root(self) -> Path:
        return self._paths_or_raise().project_root

    @property
    def _host_workspace(self) -> Path:
        return self._paths_or_raise().host_workspace

    @property
    def _tools_src_dir(self) -> Path:
        return self._paths_or_raise().tools_src_dir

    @property
    def _workspace_tools_dir(self) -> Path:
        return self._paths_or_raise().workspace_tools_dir

    @property
    def _logs_dir(self) -> Path:
        return self._paths_or_raise().logs_dir

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            self._client.close()

    def check_availability(self) -> bool:
        """Check if the Docker sandbox is available.

        Returns:
            bool: True if Docker is available and responsive, False otherwise

        Example:
            >>> sandbox = DockerSandbox()
            >>> sandbox.check_availability()
            True  # If Docker is running
            False  # If Docker is not available
        """
        if docker is None:
            logger.warning("Docker SDK is not installed")
            return False

        client = None
        try:
            client = docker.from_env()
            is_available = client.ping()
            if is_available:
                version = client.version()
                logger.info(
                    "Docker is available (Version: %s, API: %s)",
                    version.get("Version", "unknown"),
                    version.get("ApiVersion", "unknown"),
                )
            return is_available
        except Exception as e:
            logger.warning("Failed to connect to Docker", error=str(e))
            return False
        finally:
            if client:
                try:
                    client.close()
                except Exception as e:
                    logger.warning(
                        "Error closing Docker client", error=str(e), exc_info=True
                    )

    async def _discover_existing_containers(self) -> None:
        """Discover and track existing locca-sandbox containers asynchronously."""
        if not self._client:
            return

        try:
            # Find all running containers with our image
            containers = self._client.containers.list(
                filters={"ancestor": self.image, "status": "running"},
                ignore_removed=True,
            )

            # Track discovered containers using their names
            for container in containers:
                # Extract session_id from container name (format: locca-sandbox-{session_id})
                if container.name.startswith("locca-sandbox-"):
                    session_id = container.name[len("locca-sandbox-") :]
                    self._containers[session_id] = container
                    logger.info(
                        f"Discovered existing container: {container.id} (Session: {session_id})"
                    )
                else:
                    logger.warning(
                        f"Found container with unexpected name format: {container.name}"
                    )

        except Exception as e:
            logger.error(
                "Error discovering existing containers", error=str(e), exc_info=True
            )

    async def _ensure_initialized(self) -> None:
        """Ensure the sandbox is properly initialized."""
        if not self._initialized:
            await self.initialize()

    async def initialize(self) -> None:
        """Initialize the Docker client and discover existing containers."""
        if self._initialized:
            return

        try:
            await self._ensure_client()
            await self._discover_existing_containers()
            await self.ensure_directories()

            self._initialized = True
        except Exception as e:
            logger.error(
                "Failed to initialize Docker sandbox", error=str(e), exc_info=True
            )
            raise

    async def _ensure_client(self) -> None:
        """Ensure Docker client is initialized."""
        if self._client is None:
            try:
                self._client = docker.from_env()
                self._client.ping()
            except Exception as e:
                logger.error(
                    "Failed to initialize Docker client", error=str(e), exc_info=True
                )
                raise SandboxRuntimeError(
                    "Docker is not running or not installed"
                ) from e

    async def ensure_directories(self) -> SandboxPaths:
        """Ensure directories exist and cache resolved paths."""
        if self._state.paths:
            return self._state.paths

        project_root = self._path_manager.get_project_root()
        host_workspace = project_root / ".sandbox_workspace"
        host_workspace.mkdir(parents=True, exist_ok=True)

        tools_src_dir = self._path_manager.resolve_path("@tools/sandbox_tools")

        workspace_tools_dir = host_workspace / "sandbox_tools"
        workspace_tools_dir.mkdir(parents=True, exist_ok=True)

        logs_dir = self._resolve_logs_directory(host_workspace)

        paths = SandboxPaths(
            project_root=project_root,
            host_workspace=host_workspace,
            tools_src_dir=tools_src_dir,
            workspace_tools_dir=workspace_tools_dir,
            logs_dir=logs_dir,
        )
        self._state.paths = paths
        return paths

    def _resolve_logs_directory(self, host_workspace: Path) -> Path:
        """Resolve and ensure the logs directory within the workspace."""
        logging_config = getattr(self.config, "logging", None)
        configured_dir = (
            getattr(logging_config, "directory", "logs") if logging_config else "logs"
        )
        safe_dir = configured_dir.lstrip("/\\") or "logs"
        logs_dir = host_workspace / safe_dir
        logs_dir.mkdir(mode=0o775, exist_ok=True, parents=True)
        return logs_dir

    def _should_reuse_running_container(
        self, session_id: str, _persistence: bool
    ) -> bool:
        """Determine if a running container should be reused for the session."""
        if session_id == "default":
            return False
        return True

    async def _reuse_tracked_container(
        self, session_id: str, persistence: bool
    ) -> Container | None:
        """Return an already tracked container if it can be reused."""
        container = self._containers.get(session_id)
        if not container:
            return None

        try:
            container.reload()
        except Exception:
            self._containers.pop(session_id, None)
            return None

        status = getattr(container, "status", "")
        if status != "running":
            self._containers.pop(session_id, None)
            return None

        if not self._should_reuse_running_container(session_id, persistence):
            await self._cleanup_container(session_id, container, remove=True)
            return None

        return container

    async def _cleanup_container(
        self, session_id: str, container: Container, remove: bool
    ) -> None:
        """Stop and optionally remove a container, ensuring internal tracking is cleared.

        Args:
            session_id: ID of the session being cleaned up
            container: The container to clean up
            remove: Whether to remove the container and its resources
        """
        if not container:
            return

        try:
            await self.stop_container(
                container.id, remove=remove, session_id=session_id
            )
        except Exception as e:
            logger.warning(
                f"Failed to clean up container {container.id}",
                error=str(e),
                exc_info=True,
            )
        finally:
            # Clean up the session-specific IPC directory
            if session_id and session_id != "default":
                try:
                    ipc_dir = self._host_workspace / "ipc" / f"locca_ipc_{session_id}"
                    if ipc_dir.exists():
                        import shutil

                        try:
                            shutil.rmtree(ipc_dir)
                            logger.debug(f"Cleaned up IPC directory: {ipc_dir}")
                        except Exception as e:
                            logger.warning(
                                f"Failed to clean up IPC directory {ipc_dir}",
                                error=str(e),
                                exc_info=True,
                            )

                    # Also clean up any files in the container's /workspace/ipc directory
                    if container and container.status == "running":
                        try:
                            # Remove all files in the session IPC directory
                            ipc_path = f"/workspace/ipc/locca_ipc_{session_id}"
                            cmd = f"rm -rf {ipc_path}/requests/* {ipc_path}/responses/*"
                            container.exec_run(cmd, user="root")
                        except Exception as e:
                            logger.debug(
                                "Failed to clean container's IPC files",
                                error=str(e),
                                exc_info=True,
                            )

                except Exception as e:
                    logger.warning(
                        f"Failed to clean up IPC directory for session {session_id}",
                        error=str(e),
                        exc_info=True,
                    )

            # Ensure container is removed from tracking
            self._containers.pop(session_id, None)

    def _validate_persistent_capacity(self) -> None:
        """Ensure the number of persistent sessions does not exceed the configured limit."""
        persistent_count = sum(1 for cid in self._containers if cid != "default")
        if persistent_count >= self.max_sessions:
            raise SandboxRuntimeError(
                f"Max sessions ({self.max_sessions}) reached. Stop some sessions first."
            )

    async def _ensure_image_ready(self) -> None:
        """Check that the sandbox image exists and rebuild if necessary."""
        if not self._client:
            raise SandboxRuntimeError("Docker client is not initialized")

        try:
            self._client.images.get(self.image)
            if self.auto_build and self._has_docker_changes():
                logger.info(f"Docker context changed, rebuilding image {self.image}...")
                await self._build_image()
        except NotFound as e:
            if self.auto_build:
                logger.info(f"Image {self.image} not found. Building...")
                await self._build_image()
            else:
                raise SandboxRuntimeError(
                    f"Image {self.image} not found and auto_build is False."
                ) from e

    def _build_volume_mounts(self, session_id: str = "") -> dict[str, dict[str, str]]:
        """Prepare volume bindings for the container."""
        volumes = {
            str(self._host_workspace): {"bind": "/workspace", "mode": "rw"},
            str(self._logs_dir): {"bind": "/workspace/logs", "mode": "rw"},
            str(
                self._project_root
                / "src"
                / "local_coding_assistant"
                / "sandbox"
                / "guest"
            ): {
                "bind": "/agent",
                "mode": "ro",
            },
            str(self._tools_src_dir): {"bind": "/tools/sandbox_tools", "mode": "ro"},
        }

        return volumes

    def _get_network_mode(self) -> str:
        """Return the appropriate network mode."""
        return "bridge" if self.network_enabled else "none"

    def _determine_container_parameters(
        self, session_id: str, persistence: bool
    ) -> tuple[str, list[str] | None]:
        """Determine container name and command based on persistence."""
        if persistence:
            agent_path = "/agent/agent.py"
            cmd = [
                "python3",
                agent_path,
                "--daemon",
                "--timeout",
                str(self.timeout),
                "--ipc-dir",
                f"/workspace/ipc/locca_ipc_{session_id}",
            ]
            return f"locca-sandbox-{session_id}", cmd

        name = f"locca-sandbox-ephemeral-{int(time.time())}"
        return name, None

    def _build_logging_config(self, session_id: str) -> dict[str, Any]:
        """Assemble logging configuration for the container environment."""
        if not self.config:
            return {}

        logging_config = getattr(self.config, "logging", None)
        if not logging_config:
            return {}

        return {
            "level": logging_config.level,
            "console": logging_config.console,
            "file": logging_config.file,
            "directory": logging_config.directory,
            "file_name": logging_config.file_name.format(session_id=session_id),
            "max_size": logging_config.max_size,
            "backup_count": logging_config.backup_count,
        }

    def _build_environment(
        self, logging_config: dict[str, Any], session_id: str
    ) -> dict[str, str]:
        """Create environment variables for the container."""
        return {
            "PYTHONPATH": "/agent:/tools:/workspace",
            "CONTAINER_TIMEOUT": str(self.timeout),
            "IPC_DIR": f"/workspace/ipc/locca_ipc_{session_id}",
            "LOGGING_CONFIG": json.dumps(logging_config),
            "PYTHONIOENCODING": "utf-8",
        }

    async def _reuse_existing_named_container(
        self,
        session_id: str,
        name: str,
        persistence: bool,
    ) -> Container | None:
        """Reuse an existing Docker container by name if appropriate."""
        if not self._client:
            return None

        try:
            existing = self._client.containers.get(name)
        except NotFound:
            return None
        except Exception as e:
            logger.warning(
                f"Failed to inspect existing container {name}",
                error=str(e),
                exc_info=True,
            )
            return None

        try:
            existing.reload()
        except Exception as e:
            logger.debug(
                f"Failed to reload container {name}", error=str(e), exc_info=True
            )
            return None

        status = getattr(existing, "status", "")
        if status == "running":
            if not self._should_reuse_running_container(session_id, persistence):
                await self._cleanup_container(session_id, existing, remove=True)
                return None

            self._containers[session_id] = existing
            logger.info(
                f"Reusing existing sandbox container {existing.short_id} (Session: {session_id}, Persistent: {persistence})"
            )
            return existing

        try:
            existing.remove(force=True)
        except Exception as e:
            logger.debug(
                f"Failed to remove stopped container {name}",
                error=str(e),
                exc_info=True,
            )

        return None

    def _create_container(
        self,
        name: str,
        command: list[str] | None,
        volumes: dict[str, dict[str, str]],
        network_mode: str,
        environment: dict[str, str],
        persistence: bool,
    ) -> Container:
        """Create and start a new Docker container."""
        if not self._client:
            raise SandboxRuntimeError("Docker client is not initialized")

        # Filter out None values from volumes (happens when session_id is empty)
        volumes = {k: v for k, v in volumes.items() if k is not None}

        # Ensure the container starts as root so the entrypoint can set up permissions
        return self._client.containers.run(
            self.image,
            name=name,
            command=command,
            detach=True,
            tty=True,
            mem_limit=self.mem_limit,
            cpu_quota=self.cpu_quota,
            network_mode=network_mode,
            volumes=volumes,
            environment=environment,
            remove=not persistence,
            working_dir="/workspace",
            user="root",  # Start as root to allow entrypoint to set permissions
        )

    async def start_container(
        self, session_id: str, persistence: bool = False
    ) -> Container:
        """Start a sandbox container for a specific session."""
        await self._ensure_initialized()

        reused = await self._reuse_tracked_container(session_id, persistence)
        if reused:
            return reused

        if persistence:
            self._validate_persistent_capacity()

        await self._ensure_image_ready()

        volumes = self._build_volume_mounts(session_id)
        network_mode = self._get_network_mode()
        name, command = self._determine_container_parameters(session_id, persistence)
        logging_config = self._build_logging_config(session_id)
        environment = self._build_environment(logging_config, session_id)

        reused_named = await self._reuse_existing_named_container(
            session_id, name, persistence
        )
        if reused_named:
            return reused_named

        try:
            container = self._create_container(
                name=name,
                command=command,
                volumes=volumes,
                network_mode=network_mode,
                environment=environment,
                persistence=persistence,
            )
            self._containers[session_id] = container
            logger.info(
                f"Started sandbox container {container.short_id} (Session: {session_id}, Persistent: {persistence})"
            )
            return container
        except Exception as e:
            logger.error(
                "Failed to start sandbox container", error=str(e), exc_info=True
            )
            raise

    async def start(self) -> None:
        """Start default ephemeral container (compatibility)."""
        await self.start_container("default", persistence=False)

    def _find_session_id_by_container(self, container_id: str) -> str:
        """Return the session id associated with a tracked container id."""
        for session, tracked in self._containers.items():
            if getattr(tracked, "id", None) == container_id:
                return session
        return ""

    def _pop_tracked_container(
        self, session_id: str, container_id: str
    ) -> tuple[str, Container | None]:
        """Remove and return the tracked container for the given identifiers."""
        if not session_id:
            resolved_session = self._find_session_id_by_container(container_id)
        else:
            resolved_session = session_id

        container = (
            self._containers.pop(resolved_session, None) if resolved_session else None
        )
        return resolved_session, container

    def _stop_container_once(
        self,
        container: Container,
        container_id: str,
        session_id: str | None,
        remove: bool,
    ) -> bool:
        """Attempt a single stop/remove operation, returning True if complete."""
        try:
            container.reload()
            status = getattr(container, "status", "")
            logger.debug(f"Container {container_id} status: {status}")
        except (NotFound, ImageNotFound):
            logger.debug(
                f"Container {container_id} not found, assuming already removed"
            )
            return True
        except Exception as exc:  # Reload failed for other reasons
            logger.debug(
                f"Failed to reload container {container_id}",
                error=str(exc),
                exc_info=True,
            )
            return False

        if status in {"exited", "dead"}:
            if remove:
                logger.debug(
                    f"Removing container {container_id} (Session: {session_id})"
                )
                container.remove(force=True)
                logger.info(f"Removed container {container_id} (Session: {session_id})")
            return True

        logger.debug(f"Stopping container {container_id} (Session: {session_id})")
        container.stop(timeout=5)

        if remove:
            logger.debug(f"Removing container {container_id} (Session: {session_id})")
            container.remove(force=True)
            logger.info(
                f"Stopped and removed session {session_id} (Container {container_id})"
            )
        else:
            logger.info(f"Stopped session {session_id} (Container {container_id})")
        return True

    async def _gather_metrics(
        self, container: Container | None, start_stats: dict[str, Any] | None
    ) -> list[ResourceMetric]:
        if not container or not start_stats:
            return []

        try:
            end_stats = await self._get_container_stats(container)
        except Exception as exc:
            logger.warning(
                "Failed to collect container stats", error=str(exc), exc_info=True
            )
            return []

        return self._create_resource_metrics(start_stats, end_stats)

    async def _add_metrics_to_response(
        self,
        response: SandboxExecutionResponse,
        container: Container | None,
        start_stats: dict[str, Any] | None,
    ) -> None:
        metrics = await self._gather_metrics(container, start_stats)

        for metric in metrics:
            response.add_system_metric(metric)

    def _build_execution_response(
        self,
        response_data: dict[str, Any],
        duration: float,
        stderr_fallback: str = "",
        return_code_default: int | None = None,
    ) -> SandboxExecutionResponse:
        response = SandboxExecutionResponse(
            success=response_data.get("success", False),
            result=response_data.get("result"),
            stdout=response_data.get("stdout", ""),
            stderr=response_data.get("stderr", stderr_fallback),
            error=response_data.get("error"),
            duration=duration,
            return_code=response_data.get("return_code", return_code_default or 0),
            final_answer=response_data.get("final_answer"),
        )

        response.files_created = response_data.get("files_created", [])
        response.files_modified = response_data.get("files_modified", [])
        return response

    def _extract_resource_metrics(
        self, call_data: dict[str, Any]
    ) -> list[ResourceMetric]:
        metrics: list[ResourceMetric] = []

        def add_metric_if_exists(
            stats: dict[str, Any],
            stat_name: str,
            metric_name: str,
            metric_type: ResourceType,
            unit: str,
            multiplier: float = 1.0,
        ) -> None:
            if stat_name in stats:
                metrics.append(
                    ResourceMetric(
                        type=metric_type,
                        name=metric_name,
                        value=stats[stat_name] * multiplier,
                        unit=unit,
                        timestamp=datetime.fromtimestamp(
                            call_data.get("timestamp", time.time()), tz=UTC
                        ),
                    )
                )

        end_stats = call_data.get("end_stats", {})
        add_metric_if_exists(
            end_stats, "cpu_percent", "cpu_usage", ResourceType.CPU, "percent"
        )
        add_metric_if_exists(
            end_stats,
            "memory_rss_mb",
            "memory_usage",
            ResourceType.MEMORY,
            "bytes",
            1024 * 1024,
        )
        add_metric_if_exists(
            end_stats,
            "read_bytes_per_sec",
            "read_bytes_per_sec",
            ResourceType.DISK,
            "bytes/second",
        )
        add_metric_if_exists(
            end_stats,
            "write_bytes_per_sec",
            "write_bytes_per_sec",
            ResourceType.DISK,
            "bytes/second",
        )

        delta_stats = call_data.get("delta_stats", {})
        add_metric_if_exists(
            delta_stats, "cpu_delta", "cpu_delta", ResourceType.CPU, "percent"
        )
        add_metric_if_exists(
            delta_stats,
            "memory_delta_mb",
            "memory_delta",
            ResourceType.MEMORY,
            "bytes",
            1024 * 1024,
        )

        return metrics

    def _process_tool_call_metrics(
        self, response: SandboxExecutionResponse, metrics_per_tool_call: dict[str, Any]
    ) -> None:
        if not metrics_per_tool_call:
            return

        for call_data in metrics_per_tool_call.get("tool_calls", []):
            try:
                tool_call = ToolCallMetric(
                    tool_name=call_data.get("tool_name", "unknown"),
                    call_id=call_data.get("call_id", str(uuid.uuid4())),
                    start_time=datetime.fromisoformat(
                        call_data.get("start_time", datetime.now(UTC).isoformat())
                    ),
                    end_time=datetime.fromisoformat(
                        call_data.get("end_time", datetime.now(UTC).isoformat())
                    ),
                    duration=call_data.get("duration", 0.0),
                    success=call_data.get("success", False),
                    error=call_data.get("error"),
                )

                for metric in self._extract_resource_metrics(call_data):
                    tool_call.resource_metrics.append(metric)

                response.tool_calls.append(tool_call)
            except Exception as exc:
                logger.warning(
                    "Failed to process tool call metrics", error=str(exc), exc_info=True
                )

    def _get_ipc_paths(self, session_id: str) -> tuple[Path, Path]:
        """Get the request and response file paths for a session.

        Args:
            session_id: The session ID

        Returns:
            Tuple of (request_file_path, response_file_path)
        """
        # Use the session-specific IPC directory under /workspace/ipc
        # The entrypoint script will ensure proper permissions on these directories
        ipc_path = self._host_workspace / "ipc" / f"locca_ipc_{session_id}"

        request_path = ipc_path / "requests"
        response_path = ipc_path / "responses"

        # The entrypoint script will create these directories with proper permissions
        # We still check and create them here as a fallback
        request_path.mkdir(parents=True, exist_ok=True)
        response_path.mkdir(parents=True, exist_ok=True)

        req_id = str(uuid.uuid4())
        request_file = request_path / f"{req_id}.json"
        response_file = response_path / f"{req_id}.json"
        return request_file, response_file

    async def _execute_persistent_request(
        self,
        container: Container,
        request: SandboxExecutionRequest,
        start_stats: dict[str, Any] | None,
    ) -> SandboxExecutionResponse:
        request_file, response_file = self._get_ipc_paths(request.session_id)

        try:
            with open(request_file, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "code": request.code,
                        "session_id": request.session_id,
                        "timeout": request.timeout or self.timeout,
                    },
                    file,
                )

            start_time = time.time()
            timeout = request.timeout or self.timeout

            while time.time() - start_time < timeout:
                if not response_file.exists():
                    await asyncio.sleep(0.1)
                    continue

                try:
                    with open(response_file, encoding="utf-8") as file:
                        response_data = json.load(file)
                    duration = time.time() - start_time

                    metrics_per_tool_call = response_data.get(
                        "metrics_per_tool_call", {}
                    )
                    response = self._build_execution_response(response_data, duration)
                    self._process_tool_call_metrics(response, metrics_per_tool_call)
                    await self._add_metrics_to_response(
                        response, container, start_stats
                    )
                    return response
                except json.JSONDecodeError as exc:
                    raise SandboxOutputFormatError(
                        "Failed to parse response from sandbox",
                        stderr="Invalid response format from sandbox",
                    ) from exc
                except Exception as exc:
                    raise SandboxRuntimeError(
                        f"Error reading response: {exc!s}"
                    ) from exc
                finally:
                    with suppress(Exception):
                        response_file.unlink()

            raise SandboxTimeoutError(
                "Timed out waiting for response from sandbox", return_code=124
            )
        finally:
            with suppress(Exception):
                request_file.unlink()

    async def _execute_ephemeral_request(
        self,
        container: Container,
        request: SandboxExecutionRequest,
        start_stats: dict[str, Any] | None,
    ) -> SandboxExecutionResponse:
        payload = {"code": request.code, "session_id": request.session_id}
        json_payload = json.dumps(payload)

        req_file_name = f"req_{request.session_id}_{int(time.time())}.json"
        host_req_file = self._host_workspace / req_file_name
        container_req_file = f"/workspace/{req_file_name}"

        with open(host_req_file, "w", encoding="utf-8") as file:
            file.write(json_payload)

        agent_path = "/agent/agent.py"
        cmd = ["python3", agent_path, container_req_file]

        start_time = time.time()

        try:
            _exit_code, streams = container.exec_run(cmd, demux=True)
            stdout_bytes, stderr_bytes = streams if streams else (b"", b"")
            stdout_str = (
                stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
            )
            stderr_str = (
                stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
            )

            try:
                response_data = json.loads(stdout_str or "{}")
                duration = time.time() - start_time
                response = self._build_execution_response(
                    response_data, duration, stderr_fallback=stderr_str
                )
                await self._add_metrics_to_response(response, container, start_stats)
                return response
            except json.JSONDecodeError as exc:
                raise SandboxOutputFormatError(
                    "Failed to parse sandbox response",
                    stdout=stdout_str,
                    stderr=stderr_str or "JSON Decode Error",
                ) from exc
        finally:
            with suppress(Exception):
                host_req_file.unlink()

    async def stop_container(
        self,
        container_id: str,
        remove: bool = True,
        max_retries: int = 3,
        retry_delay: float = 0.5,
        session_id: str = "",
    ) -> None:
        """
        Stop a specific container.

        Args:
            container_id: ID of the container to stop
            remove: If True, also remove the container (default: True)
            max_retries: Maximum number of retries for container stop (default: 3)
            retry_delay: Delay between retries in seconds (default: 0.5)
            session_id: ID of the session to stop
        """
        await self._ensure_initialized()

        session_id, container = self._pop_tracked_container(session_id, container_id)
        if not container:
            logger.warning(f"Can't stop container. Session {session_id} not found")
            return

        for attempt in range(max_retries):
            if attempt:  # Skip sleep on first attempt
                await asyncio.sleep(retry_delay * (attempt + 1))

            try:
                if self._stop_container_once(
                    container, container_id, session_id, remove
                ):
                    return
            except APIError as err:
                if err.status_code == 409 and "already in progress" in str(err):
                    logger.debug(
                        f"Container {container_id} removal already in progress"
                    )
                    continue
                raise
            except Exception as exc:
                logger.debug(
                    f"Attempt {attempt + 1} failed for container {container_id}",
                    error=str(exc),
                )
                if attempt == max_retries - 1:
                    logger.error(
                        f"Failed to stop container {container_id} after {max_retries} attempts",
                        error=str(exc),
                    )
                    if remove and self._client:
                        with suppress(Exception):
                            self._client.containers.get(container_id).remove(
                                force=True, v=True
                            )

    async def stop_session(self, session_id: str, remove: bool = True) -> None:
        """Stop a specific session.

        Args:
            session_id: ID of the session to stop
            remove: If True, also remove the container (default: True)
        """
        await self._ensure_initialized()
        container = self._containers.get(session_id)

        container_id = container.id if container else "unknown"
        await self.stop_container(container_id, remove=remove, session_id=session_id)

    async def stop(self, remove_containers: bool = True) -> None:
        """Stop all sandbox containers.

        Args:
            remove_containers: If True, remove the containers after stopping (default: True)
        """
        await self._ensure_initialized()
        # Create a copy of the container list to avoid modification during iteration
        session_ids = list(self._containers.keys())
        for session_id in session_ids:
            try:
                logger.debug(f"Stopping session {session_id}")
                await self.stop_session(session_id, remove=remove_containers)
            except Exception as e:
                logger.error(
                    f"Error stopping session {session_id}", error=str(e), exc_info=True
                )
                # Continue stopping other containers even if one fails

        if self._client:
            self._client.close()

    async def execute_shell(
        self,
        command: str,
        session_id: str = "default",
        timeout: int = 30,
        env: dict[str, str] | None = None,
        working_dir: str | None = None,
        user: str | None = None,
        privileged: bool = False,
    ) -> SandboxExecutionResponse:
        """Execute a shell command in the sandbox.

        Args:
            command: The shell command to execute
            session_id: Session ID for the execution
            timeout: Timeout in seconds
            env: Environment variables to set for the command
            working_dir: Working directory for the command
            user: Username or UID to run the command as
            privileged: Whether to run the command in privileged mode

        Returns:
            SandboxExecutionResponse with command output and status
        """
        logger.debug(
            f"Executing shell command in sandbox with session {session_id}",
            command=command,
        )

        # Security check
        try:
            self.security_manager.validate_command(command)
        except SandboxSecurityError as exc:
            response = SandboxExecutionResponse()
            response.success = False
            response.error = str(exc)
            response.stderr = str(exc)
            response.return_code = 1
            response.finalize()
            return response

        response = SandboxExecutionResponse()
        container: Container | None = None

        try:
            container = await self.start_container(session_id, persistence=False)
            start_stats = (
                await self._get_container_stats(container) if container else None
            )

            loop = asyncio.get_running_loop()
            env_vars = env or {}

            def _run_command(cmd: str):
                return container.exec_run(
                    ["sh", "-c", cmd],
                    environment=env_vars,
                    workdir=working_dir,
                    user=user,
                    privileged=privileged,
                    demux=True,
                )

            try:
                exit_code, output = await asyncio.wait_for(
                    loop.run_in_executor(None, _run_command, command),
                    timeout=timeout,
                )

                stdout = (
                    output[0].decode("utf-8", errors="replace") if output[0] else ""
                )
                stderr = (
                    output[1].decode("utf-8", errors="replace") if output[1] else ""
                )

                await self._add_metrics_to_response(response, container, start_stats)

                response.success = exit_code == 0
                response.return_code = exit_code
                response.stdout = stdout
                response.stderr = stderr

                if exit_code != 0:
                    response.error = f"Command failed with exit code {exit_code}"

            except TimeoutError as exc:
                await self._add_metrics_to_response(response, container, start_stats)
                raise SandboxTimeoutError(
                    f"Command timed out after {timeout} seconds", return_code=124
                ) from exc

        except SandboxSecurityError as exc:
            response.success = False
            response.error = str(exc)
            response.stderr = str(exc)
            response.return_code = 1
        except SandboxTimeoutError as exc:
            response.success = False
            response.error = str(exc)
            response.stderr = str(exc)
            response.return_code = exc.return_code or 124
        except SandboxRuntimeError as exc:
            response.success = False
            response.error = str(exc)
            response.stderr = str(exc)
            response.return_code = getattr(exc, "return_code", 1)
        except Exception as exc:
            response.success = False
            response.error = str(exc)
            response.stderr = str(exc)
            response.return_code = 1

        finally:
            if container:
                await self._cleanup_container(session_id, container, remove=True)

            response.finalize()

        return response

    def _create_resource_metrics(
        self, start_stats: dict[str, Any] | None, end_stats: dict[str, Any] | None
    ) -> list[ResourceMetric]:
        """Create resource metrics from container stats.

        Args:
            start_stats: Container stats before execution or None
            end_stats: Container stats after execution or None

        Returns:
            List of ResourceMetric objects
        """
        metrics: list[ResourceMetric] = []
        now = datetime.now(UTC)

        if not start_stats or not end_stats:
            logger.debug("Missing container stats for resource metrics")
            return metrics

        try:
            # CPU Metrics
            if all(
                k in end_stats.get("cpu_stats", {})
                for k in ["cpu_usage", "system_cpu_usage", "online_cpus"]
            ):
                cpu_stats = end_stats["cpu_stats"]
                precpu_stats = end_stats.get("precpu_stats", {})

                # Calculate CPU usage percentage
                cpu_delta = cpu_stats["cpu_usage"]["total_usage"] - precpu_stats.get(
                    "cpu_usage", {}
                ).get("total_usage", 0)
                system_delta = cpu_stats["system_cpu_usage"] - precpu_stats.get(
                    "system_cpu_usage", 0
                )
                num_cores = cpu_stats["online_cpus"]

                if system_delta > 0 and cpu_delta > 0:
                    cpu_percent = (cpu_delta / system_delta) * num_cores * 100.0
                    metrics.append(
                        ResourceMetric(
                            type=ResourceType.CPU,
                            name="cpu_usage",
                            value=round(cpu_percent, 2),
                            unit="percent",
                            timestamp=now,
                        )
                    )

            # Memory Metrics
            if "memory_stats" in end_stats:
                mem_stats = end_stats["memory_stats"]
                if "usage" in mem_stats:
                    metrics.append(
                        ResourceMetric(
                            type=ResourceType.MEMORY,
                            name="memory_usage",
                            value=mem_stats["usage"],
                            unit="bytes",
                            timestamp=now,
                        )
                    )

                if "limit" in mem_stats:
                    metrics.append(
                        ResourceMetric(
                            type=ResourceType.MEMORY,
                            name="memory_limit",
                            value=mem_stats["limit"],
                            unit="bytes",
                            timestamp=now,
                        )
                    )

            # Throttling metrics
            if "throttling_data" in end_stats.get("cpu_stats", {}):
                throttling = end_stats["cpu_stats"]["throttling_data"]
                metrics.extend(
                    [
                        ResourceMetric(
                            type=ResourceType.CPU,
                            name="throttled_periods",
                            value=throttling.get("throttled_periods", 0),
                            unit="count",
                            timestamp=now,
                        ),
                        ResourceMetric(
                            type=ResourceType.CPU,
                            name="throttled_time",
                            value=throttling.get("throttled_time", 0),
                            unit="nanoseconds",
                            timestamp=now,
                        ),
                    ]
                )

        except Exception as e:
            logger.warning(
                "Error creating resource metrics", error=str(e), exc_info=True
            )

        return metrics

    async def execute(
        self, request: SandboxExecutionRequest
    ) -> SandboxExecutionResponse:
        """Execute code in the sandbox.

        Args:
            request: The execution request containing code and parameters

        Returns:
            SandboxExecutionResponse with execution results and metrics
        """
        logger.debug(f"Executing code in sandbox with session {request.session_id}")

        response = SandboxExecutionResponse()
        container: Container | None = None

        try:
            self.security_manager.validate_code(request.code)
        except SandboxSecurityError as exc:
            response.success = False
            response.error = str(exc)
            response.finalize()
            return response

        try:
            container = await self.start_container(
                request.session_id, request.persistence
            )
            start_stats = (
                await self._get_container_stats(container) if container else None
            )

            if request.persistence:
                response = await self._execute_persistent_request(
                    container, request, start_stats
                )
            else:
                response = await self._execute_ephemeral_request(
                    container, request, start_stats
                )

        except SandboxSecurityError as exc:
            response = SandboxExecutionResponse(
                success=False,
                error=str(exc),
            )
        except SandboxTimeoutError as exc:
            response = SandboxExecutionResponse(
                success=False,
                error=str(exc),
                return_code=exc.return_code or 124,
            )
        except SandboxRuntimeError as exc:
            logger.error("Execution failed", error=str(exc), exc_info=True)
            response = SandboxExecutionResponse(
                success=False,
                error=str(exc),
                stderr=traceback.format_exc(),
                return_code=getattr(exc, "return_code", 1),
            )
        finally:
            if not request.persistence and container:
                await self._cleanup_container(
                    request.session_id, container, remove=True
                )

            response.finalize()

        return response

    def _get_docker_context_files(self) -> list[Path]:
        """Get a list of files that are part of the Docker build context."""
        docker_dir = self._path_manager.resolve_path(
            "@project/src/local_coding_assistant/sandbox"
        )
        return [
            docker_dir / "Dockerfile",
            docker_dir / "docker-entrypoint.sh",
        ]

    def _has_docker_changes(self) -> bool:
        """Check if any Docker-related files have been modified since the image was built."""
        try:
            image = self._client.images.get(self.image)
            # image.attrs['Created'] is an ISO 8601 string (e.g., "2025-12-19T19:23:00.123456Z")
            created_str = image.attrs["Created"]

            # Python 3.11+ natively handles the 'Z' suffix.
            # For older versions (3.7-3.10), use .replace('Z', '+00:00')
            try:
                image_created_dt = datetime.fromisoformat(created_str)
            except ValueError:
                # Fallback for older Python versions to handle 'Z'
                image_created_dt = datetime.fromisoformat(
                    created_str.replace("Z", "+00:00")
                )

            image_created_ts = image_created_dt.timestamp()

            for file_path in self._get_docker_context_files():
                if not file_path.exists():
                    continue

                # st_mtime is a Unix timestamp
                file_mtime = file_path.stat().st_mtime
                if file_mtime > image_created_ts:
                    logger.info(f"Docker context file modified: {file_path}")
                    return True
            return False
        except Exception as e:
            logger.debug(
                "Could not check for Docker changes", error=str(e), exc_info=True
            )
            return True  # If we can't check, assume changes exist

    async def _build_image(self) -> None:
        """Build the sandbox image."""
        dockerfile_path = self._path_manager.resolve_path(
            "@project/src/local_coding_assistant/sandbox"
        )
        logger.info(f"Building image from {dockerfile_path}")

        # Build with default caching behavior
        self._client.images.build(
            path=str(dockerfile_path),
            tag=self.image,
            rm=True,  # Remove intermediate containers
        )
        logger.info(f"Built image {self.image}")

    async def _get_container_stats(self, container):
        """Get container statistics.

        Args:
            container: The container to get stats for

        Returns:
            Dictionary containing container stats or None if container is None
        """
        if not container:
            return None

        def _get_stats():
            # Get stats as a single dictionary (not a generator)
            stats = container.stats(stream=False)
            # If stats is a generator, get the first item
            if hasattr(stats, "__next__"):
                return next(stats)
            return stats

        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, _get_stats)
        except Exception as e:
            logger.warning("Failed to get container stats", error=str(e), exc_info=True)
            return None
