"""Manager for sandbox instances."""

from pathlib import Path
from typing import TYPE_CHECKING

from local_coding_assistant.utils.logging import get_logger

from .base import ISandbox
from .docker_sandbox import DockerSandbox
from .security import SecurityManager

if TYPE_CHECKING:
    from local_coding_assistant.config.path_manager import PathManager
    from local_coding_assistant.config.schemas import SandboxConfig
    from local_coding_assistant.core.protocols import IConfigManager

logger = get_logger("sandbox.manager")


class SandboxManager:
    """Manages the lifecycle and configuration of the sandbox."""

    def __init__(self, config_manager: "IConfigManager"):
        self._config_manager = config_manager
        self.path_manager: PathManager = config_manager.path_manager
        self._sandbox: ISandbox | None = None
        self.security_manager: SecurityManager | None = None

    @property
    def config(self) -> "SandboxConfig":
        """Get the current sandbox config with all overrides applied."""
        return self._config_manager.global_config.sandbox

    @property
    def is_enabled(self) -> bool:
        """Check if sandbox is enabled."""
        return self.config.enabled

    def get_sandbox(self) -> ISandbox:
        """Get the configured sandbox instance."""
        if self._sandbox is None:
            self._sandbox = self._create_sandbox()
        return self._sandbox

    def _create_sandbox(self) -> ISandbox:
        """Create a new sandbox instance based on configuration."""
        if not self.config.enabled:
            # If disabled, we could return a dummy sandbox or raise an error.
            # For now, we'll return the DockerSandbox but it might not be started if logic checks enabled flag.
            # However, DockerSandbox logic assumes it should run if methods are called.
            # Let's assume the caller checks config.enabled before calling get_sandbox usually,
            # or we return a NullSandbox.
            # But the requirement is to use DockerSandbox.
            logger.warning("Sandbox is disabled, but DockerSandbox will be used.")
            pass

        self.security_manager = SecurityManager(
            allowed_imports=self.config.allowed_imports,
            blocked_patterns=self.config.blocked_patterns,
            blocked_shell_commands=self.config.blocked_shell_commands,
        )

        # Get the session_timeout from config or use the same as timeout if not specified
        session_timeout = getattr(self.config, "session_timeout", None)

        return DockerSandbox(
            image=self.config.image,
            timeout=session_timeout if session_timeout is not None else 300,
            mem_limit=self.config.memory_limit,
            cpu_quota=int(self.config.cpu_limit * 100000),
            network_enabled=self.config.network_enabled,
            security_manager=self.security_manager,
            path_manager=self.path_manager,
            max_sessions=getattr(self.config, "max_sessions", 5),
            config=self.config,  # Pass the entire config object
        )

    async def start(self) -> None:
        """Start the sandbox if enabled."""
        if self.config.enabled:
            sandbox = self.get_sandbox()
            await sandbox.start()

    async def stop(self) -> None:
        """Stop the sandbox."""
        if self._sandbox:
            await self._sandbox.stop()

    def ensure_availability(self) -> bool:
        """Ensure the sandbox is available and update config if not.

        Returns:
            bool: True if sandbox is available, False otherwise
        """
        try:
            sandbox = self.get_sandbox()
            if hasattr(sandbox, "check_availability"):
                is_available = sandbox.check_availability()
                if not is_available:
                    logger.warning("Sandbox is not available, disabling in config")
                    # Update the config to disable the sandbox
                    self._config_manager.set_session_overrides(
                        {"sandbox.enabled": False}
                    )
                    return False
                return True
            return False
        except Exception as e:
            logger.warning(
                "Error checking sandbox availability", error=str(e), exc_info=True
            )
            # Update the config to disable the sandbox
            self._config_manager.set_session_overrides({"sandbox.enabled": False})
            return False

    def get_workspace_dir(self) -> "Path":
        """Return the host workspace directory used for sandbox mounts."""
        workspace = self.path_manager.get_project_root() / ".sandbox_workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace
