"""Path management for Local Coding Assistant."""

import os
from pathlib import Path

from local_coding_assistant.utils.logging import get_logger

logger = get_logger("config.path_manager")


class PathManager:
    """Manages file system paths for different environments."""

    def __init__(
        self,
        *,
        is_development: bool = False,
        is_testing: bool = False,
        is_production: bool | None = None,
        project_root: str | Path | None = None,
    ):
        """Initialize the path manager.

        Args:
            is_development: Whether we're in development environment
            is_testing: Whether we're in testing environment
            is_production: Whether we're in production environment (auto-detected if None)
            project_root: Optional project root path (auto-detected if None)
        """
        # Determine environment if not explicitly set
        if is_production is None:
            is_production = not (is_development or is_testing)

        self._is_development = is_development
        self._is_testing = is_testing
        self._is_production = is_production

        # Set up paths
        self._project_root = (
            Path(project_root) if project_root else self._find_project_root()
        )
        self._ensure_directories()

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self._is_development

    @property
    def is_testing(self) -> bool:
        """Check if running in test environment."""
        return self._is_testing

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self._is_production

    def _find_project_root(self) -> Path:
        """Find the project root directory."""
        # Look for pyproject.toml or .git directory to identify project root
        current = Path(__file__).parent
        for parent in [current, *current.parents]:
            if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
                return parent
        return Path.cwd()

    def _ensure_directories(self) -> None:
        """Ensure all required directories exist for the active environment."""
        paths_to_ensure: list[Path] = [
            self.get_data_dir(),
            self.get_cache_dir(),
            self.get_log_dir(),
        ]

        config_dir = self.get_config_dir()
        if config_dir:
            paths_to_ensure.append(config_dir)

        if self.is_testing or self.is_development:
            module_dir = self.get_module_dir()
            if module_dir:
                paths_to_ensure.append(module_dir)

        for directory in paths_to_ensure:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.debug("Failed to ensure directory %s: %s", directory, exc)

    def get_project_root(self) -> Path:
        """Get the project root directory."""
        return self._project_root

    def get_config_dir(self) -> Path:
        """Get the configuration directory."""
        # In development, use project's config directory
        if self.is_testing:
            return self._project_root / "tests" / "configs"
        if self.is_development:
            return self._project_root / "config"
        # In production/other, use system config directory
        return self._get_system_config_dir()

    def get_data_dir(self) -> Path:
        """Get the data directory for the current environment."""
        if self.is_testing:
            return self._project_root / "tests" / "data"
        if self.is_development:
            return self._project_root / "data"
        # production
        return self._get_system_data_dir() / "local-coding-assistant"

    def get_cache_dir(self) -> Path:
        """Get the cache directory for the current environment."""
        if self.is_testing:
            return self._project_root / "tests" / "cache"
        if self.is_development:
            return self._project_root / ".cache"
        # production
        return self._get_system_cache_dir() / "local-coding-assistant"

    def get_log_dir(self) -> Path:
        """Get the log directory for the current environment."""
        if self.is_testing:
            return self._project_root / "tests" / "logs"
        if self.is_development:
            return self._project_root / "logs"
        # production
        return self._get_system_log_dir() / "local-coding-assistant"

    def _get_system_config_dir(self) -> Path:
        """Get the system configuration directory."""
        if os.name == "nt":  # Windows
            return (
                Path(os.environ.get("APPDATA", ""))
                / "Local"
                / "LocalCodingAssistant"
                / "config"
            )
        # Unix-like
        return Path.home() / ".config" / "local-coding-assistant"

    def _get_system_data_dir(self) -> Path:
        """Get the system data directory."""
        if os.name == "nt":  # Windows
            return Path(os.environ.get("LOCALAPPDATA", "")) / "LocalCodingAssistant"
        # Unix-like
        return Path.home() / ".local" / "share"

    def _get_system_cache_dir(self) -> Path:
        """Get the system cache directory."""
        if os.name == "nt":  # Windows
            return (
                Path(os.environ.get("LOCALAPPDATA", ""))
                / "LocalCodingAssistant"
                / "Cache"
            )
        # Unix-like
        return Path.home() / ".cache"

    def _get_system_log_dir(self) -> Path:
        """Get the system log directory."""
        if os.name == "nt":  # Windows
            return (
                Path(os.environ.get("LOCALAPPDATA", ""))
                / "LocalCodingAssistant"
                / "Logs"
            )
        # Unix-like
        return Path("/var") / "log" / "local-coding-assistant"

    def get_module_dir(self) -> Path:
        """Get the module directory for the current environment."""
        if self.is_testing:
            return self._project_root / "tests" / "modules"
        elif self.is_development:
            return self._project_root / "config" / "modules"
        else:  # production
            # In production, use the site-packages directory
            import site

            # Get the first site-packages directory (usually the user's site-packages)
            site_packages = (
                site.getsitepackages()[0] if site.getsitepackages() else None
            )
            if not site_packages:
                # Fallback to a relative path if site-packages can't be determined
                return (
                    Path("lib") / "site-packages" / "local_coding_assistant" / "modules"
                )
            return Path(site_packages) / "local_coding_assistant" / "modules"

    def get_tools_dir(self) -> Path:
        """Get the tools directory for the current environment."""
        if self.is_testing:
            return self._project_root / "tests" / "tools"

        return self._project_root / "src" / "local_coding_assistant" / "tools"

    def get_sandbox_guest_dir(self) -> Path:
        """Get the sandbox guest directory path.

        Returns:
            Path to the sandbox's guest directory where the agent code is mounted in the container.
        """
        return (
            self._project_root / "src" / "local_coding_assistant" / "sandbox" / "guest"
        )

    def resolve_path(
        self,
        path: str | Path,
        *,
        ensure_parent: bool = False,
        base_dir: str | Path | None = None,
    ) -> Path:
        """Resolve a path relative to the appropriate base directory.

        Args:
            path: The path to resolve (can be relative or absolute, may include @-prefixes)
            ensure_parent: If True, ensure the parent directory exists
            base_dir: Optional base directory for resolving relative paths

        Returns:
            Resolved absolute path
        """
        path = Path(path)
        if path.is_absolute():
            return path

        path_str = str(path)
        resolved = (
            self._resolve_special_path(path_str) if path_str.startswith("@") else None
        )

        if resolved is None:
            base = Path(base_dir) if base_dir else self._project_root
            resolved = base / path

        if ensure_parent:
            resolved.parent.mkdir(parents=True, exist_ok=True)

        return resolved.resolve()

    def _resolve_special_path(self, path_str: str) -> Path | None:
        """Resolve paths with special @-prefixed locations.

        Args:
            path_str: Path string starting with @

        Returns:
            Resolved Path or None if no match found
        """
        # Map of prefix to corresponding directory method
        prefix_handlers = {
            "config": self.get_config_dir,
            "data": self.get_data_dir,
            "cache": self.get_cache_dir,
            "log": self.get_log_dir,
            "module": self.get_module_dir,
            "project": lambda: self._project_root,
            "tools": self.get_tools_dir,
        }

        # Split on the first path separator
        parts = path_str[1:].replace("\\", "/").split("/", 1)
        if len(parts) != 2:
            return None

        prefix, rest = parts
        handler = prefix_handlers.get(prefix)
        if not handler:
            return None

        return handler() / rest.replace("/", os.sep)
