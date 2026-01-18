"""Environment variable management for Local Coding Assistant."""

import os
from pathlib import Path
from typing import Any

from local_coding_assistant.config.path_manager import PathManager
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("config.env_manager")

# Module-level instance
_instance = None


def get_env_manager(**kwargs) -> "EnvManager":
    """Get the singleton instance of EnvManager.

    Args:
        **kwargs: Arguments to pass to the EnvManager constructor (only used on first call)

    Returns:
        The singleton instance of EnvManager
    """
    global _instance
    if _instance is None:
        # Create the internal _EnvManager instance
        _instance = _EnvManager(**kwargs)
    # Return a new EnvManager instance which will use the singleton via __new__
    return EnvManager()


class EnvManager:
    """Public API for environment management with singleton behavior.

    This class provides backward compatibility and delegates all calls to the
    singleton instance of _EnvManager.
    """

    _instance: "_EnvManager | None" = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "EnvManager":
        """Return the singleton instance of EnvManager."""
        if cls._instance is None or os.getenv("LOCCA_ENV") == "test":
            cls._instance = _EnvManager(*args, **kwargs)

        # Create and return an EnvManager instance that will delegate to the singleton
        instance = super().__new__(cls)
        return instance

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize is a no-op since we use __new__ for singleton behavior."""
        pass

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the singleton instance."""
        if self._instance is None:
            raise RuntimeError("Environment manager not initialized")
        return getattr(self._instance, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Delegate attribute assignment to the singleton instance."""
        if name == "_instance":
            super().__setattr__(name, value)
        elif self._instance is not None:
            setattr(self._instance, name, value)
        else:
            super().__setattr__(name, value)


class _EnvManager:
    """Manages loading and parsing environment variables for configuration.

    This is the actual implementation class. Use the module-level functions
    or the public EnvManager class to get the singleton instance.
    """

    def __init__(
        self,
        env_prefix: str = "LOCCA_",
        env_paths: list[Path | str] | None = None,
        path_manager: PathManager | None = None,
        load_env: bool = True,
    ):
        """Initialize the environment manager.

        Args:
            env_prefix: Prefix for environment variables to load (default: "LOCCA_")
            env_paths: Optional list of .env file paths to load (default: auto-detect)
            path_manager: Optional PathManager instance for path resolution
        """
        self.env_prefix = env_prefix
        self._env_loaded = False

        # Initialize PathManager with current environment state
        self.path_manager = path_manager or PathManager(
            is_development=self.is_development(),
            is_testing=self.is_testing(),
            is_production=self.is_production(),
        )
        self.env_paths: list[Path] = (
            [Path(p) for p in env_paths] if env_paths else self._get_default_env_paths()
        )

        if load_env:
            self.load_env_files()

    def reload_env_files(self) -> None:
        """Force reload environment files, even if already loaded."""
        self.load_env_files(force_reload=True)

    @classmethod
    def create(cls, *, load_env: bool = True, **kwargs) -> "_EnvManager":
        """Factory method to create and initialize an EnvManager.

        Args:
            load_env: If True, automatically load environment files
            **kwargs: Additional arguments to pass to __init__

        Returns:
            Initialized _EnvManager instance
        """
        instance = cls(**kwargs)
        if load_env:
            try:
                instance.load_env_files()
            except Exception as e:
                logger.warning(
                    "Failed to load environment files", error=str(e), exc_info=True
                )
        return instance

    def _resolve_env_paths(self) -> None:
        """Resolve all environment file paths using PathManager."""
        resolved_paths = []
        for path in self.env_paths:
            if isinstance(path, str) and ("@" in path or "~" in str(path)):
                try:
                    resolved = self.path_manager.resolve_path(path)
                    resolved_paths.append(resolved)
                    continue
                except (ValueError, OSError) as e:
                    logger.warning(
                        f"Failed to resolve path {path}", error=str(e), exc_info=True
                    )
            resolved_paths.append(Path(path) if isinstance(path, str) else path)
        self.env_paths = resolved_paths

    def with_prefix(self, key: str) -> str:
        """Add the environment prefix to a key if not already present."""
        key = key.upper()
        return key if key.startswith(self.env_prefix) else f"{self.env_prefix}{key}"

    def without_prefix(self, key: str) -> str:
        """Remove the environment prefix from a key if present."""
        if key.startswith(self.env_prefix):
            return key[len(self.env_prefix) :]
        return key

    def _get_default_env_paths(self) -> list[Path]:
        """Get default .env file paths to check based on environment.

        Loading order (first takes precedence):
        1. .env - Base settings (committed to VCS)
        2. .env.<env> - Environment-specific settings (e.g., .env.development)
        3. .env.local - Local overrides (ignored by VCS, for sensitive data)

        Environment is determined by the PathManager or LOCCA_ENV env var.
        """
        env = self.get_environment()

        # Use PathManager to get project root if available
        root_dir = self.path_manager.get_project_root()
        if not root_dir:
            # Fallback to finding project root by looking for pyproject.toml
            current_path = Path(__file__).parent
            root_dir = None
            for parent in [current_path, *list(current_path.parents)]:
                if (parent / "pyproject.toml").exists():
                    root_dir = parent
                    break
            root_dir = root_dir or Path.cwd()

        # Define the loading order (from lowest to highest priority)
        env_files = [
            "@project/.env",  # Base settings
            f"@project/.env.{env}",  # Environment-specific settings
            "@project/.env.local",  # Local overrides (highest priority)
        ]

        # Resolve paths using PathManager
        resolved_paths = []
        for env_file in env_files:
            try:
                resolved = self.path_manager.resolve_path(env_file)
                if resolved.exists():
                    resolved_paths.append(resolved)
            except (ValueError, OSError) as e:
                logger.debug(
                    f"Skipping env file {env_file}", error=str(e), exc_info=True
                )
                continue

        # Always include the root .env file if it exists, even if empty
        root_env = root_dir / ".env"
        if root_env.exists() and root_env not in resolved_paths:
            resolved_paths.insert(0, root_env)

        return resolved_paths or [root_env]

    def load_env_files(self, force_reload: bool = False) -> None:
        """Load .env files if python-dotenv is available.

        Args:
            force_reload: If True, reload environment files even if already loaded.
                         If False, do nothing if already loaded (default: False).

        Files are loaded in the following order (later files override earlier ones):
        1. .env - Base settings
        2. .env.<env> - Environment-specific settings
        3. .env.local - Local overrides (highest priority)

        Environment is determined by LOCCA_ENV env var (defaults to 'development').
        This method is idempotent - subsequent calls with force_reload=False will be no-ops.

        Raises:
            ConfigError: If .env file exists but cannot be loaded
        """
        if self._env_loaded and not force_reload:
            logger.debug("Environment files already loaded, skipping")
            return

        try:
            from dotenv import load_dotenv
        except ImportError:
            logger.warning(
                "python-dotenv not installed, skipping .env file loading. "
                "Install with `pip install python-dotenv` for .env file support."
            )
            return

        env = os.getenv("LOCCA_ENV", "development").lower()
        logger.debug("Loading environment files for '%s' environment", env)

        loaded_any = False
        for env_path in self.env_paths:
            try:
                # Resolve path using PathManager if it's a string with special prefixes
                if isinstance(env_path, str) and (
                    "@" in env_path or "~" in str(env_path)
                ):
                    env_path = self.path_manager.resolve_path(env_path)

                if env_path.exists():
                    load_dotenv(env_path, override=True)
                    loaded_any = True
                    logger.debug("Loaded environment from %s", env_path)
            except Exception as e:
                from local_coding_assistant.core.exceptions import ConfigError

                raise ConfigError(
                    f"Failed to load environment from {env_path}: {e!s}"
                ) from e

        if not loaded_any:
            logger.warning(
                "No .env files found. Using system environment variables only."
            )

        self._env_loaded = True

    def get_all_env_vars(self) -> dict[str, str]:
        """Get all environment variables with the configured prefix.

        Returns:
            Dictionary of all environment variables that start with the prefix,
            with their original case preserved.
        """
        return {k: v for k, v in os.environ.items() if k.startswith(self.env_prefix)}

    def get_config_from_env(self) -> dict[str, Any]:
        """Extract configuration from environment variables.

        Returns:
            Dictionary with configuration parsed from environment variables

        Raises:
            ConfigError: If environment variable parsing fails
        """
        config_data: dict[str, Any] = {}
        env_count = 0

        try:
            for key, value in os.environ.items():
                if key.startswith(self.env_prefix):
                    config_key = key[len(self.env_prefix) :].lower()
                    env_count += 1

                    # Convert double underscores to nested structure
                    # e.g., LLM__MODEL_NAME → {"llm": {"model_name": value}}
                    key_parts = config_key.split("__")
                    self._set_nested_value(config_data, key_parts, value)

            if env_count > 0:
                logger.debug(
                    f"Loaded {env_count} environment variables with prefix '{self.env_prefix}'"
                )

        except Exception as e:
            from local_coding_assistant.core.exceptions import ConfigError

            raise ConfigError(f"Failed to parse environment variables: {e}") from e

        return config_data

    def _set_nested_value(
        self, data: dict[str, Any], key_parts: list[str], value: str
    ) -> None:
        """Set a nested dictionary value from key parts.

        Args:
            data: Dictionary to modify
            key_parts: List of keys representing the path (e.g., ['llm', 'model_name'])
            value: Value to set (will be converted from string)
        """
        current = data

        # Navigate to the correct nested location
        for _i, part in enumerate(key_parts[:-1]):
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                # If it's not a dict, we need to handle this case
                # For now, we'll just log a warning and skip
                logger.warning(
                    f"Cannot set nested value for {'.'.join(key_parts)}: {part} is not a dictionary"
                )
                return
            current = current[part]

        # Set the final value with type conversion
        final_key = key_parts[-1]
        current[final_key] = self._convert_env_value(value)

    def _convert_env_value(self, value: str) -> Any:
        """Convert string environment variable value to appropriate type.

        Args:
            value: String value from environment

        Returns:
            Converted value (bool, int, float, or string)
        """
        # Handle boolean values
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # Handle numeric values
        try:
            # Try integer first
            if "." not in value and "e" not in value.lower():
                return int(value)
            # Try float
            return float(value)
        except ValueError:
            pass

        # Return as string (including empty strings and None)
        return value if value != "null" else None

    def get_env_path(self, filename: str = ".env.local") -> Path:
        """Get the path to an environment file, creating parent directories if needed.

        Args:
            filename: Name of the environment file (e.g., ".env.local")

        Returns:
            Path to the environment file
        """
        # First check if file exists in any of the default locations
        for path in self.env_paths:
            if path.name == filename and path.exists():
                return path

        # If not found, use the first path that matches the filename pattern
        for path in self.env_paths:
            if path.name.endswith(filename.lstrip(".")):
                path.parent.mkdir(parents=True, exist_ok=True)
                return path

        # Fallback to current working directory
        path = Path.cwd() / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def save_to_env_file(
        self,
        key: str,
        value: str,
        env_path: Path | None = None,
        quote_mode: str = "never",
    ) -> None:
        """Save a key-value pair to an environment file.

        Args:
            key: Environment variable name
            value: Value to set
            env_path: Optional path to .env file (uses default if None)
            quote_mode: How to handle quoting values ("always", "never", "auto")
        """
        from dotenv import set_key

        if env_path is None:
            env_path = self.get_env_path()

        env_path.parent.mkdir(parents=True, exist_ok=True)
        set_key(env_path, key, value, quote_mode=quote_mode)

    def remove_from_env_file(
        self, key: str, env_path: Path | None = None, quote_mode: str = "never"
    ) -> None:
        """Remove a key from an environment file.

        Args:
            key: Environment variable name to remove
            env_path: Optional path to .env file (uses default if None)
            quote_mode: Quote mode for dotenv (must match how values were saved)
        """
        from dotenv import unset_key

        if env_path is None:
            env_path = self.get_env_path()

        if env_path.exists():
            unset_key(env_path, key, quote_mode=quote_mode)

    def set_env(self, key: str, value: str) -> None:
        """Set an environment variable in the current process.

        Args:
            key: Environment variable name (without prefix)
            value: Value to set
        """
        os.environ[self.with_prefix(key)] = str(value)

    def unset_env(self, key: str) -> None:
        """Unset an environment variable in the current process.

        Args:
            key: Environment variable name to unset (without prefix)
        """
        os.environ.pop(self.with_prefix(key), None)

    def get_env(
        self, name: str, default: Any = None, prefix: str | None = None
    ) -> str | None:
        """Get an environment variable, optionally with a prefix.

        Args:
            name: The name of the environment variable
            default: Default value if the variable is not found
            prefix: Optional prefix to try before the bare variable name

        Returns:
            The value of the environment variable if found, otherwise the default value
        """
        key = name.upper()
        # First try with prefix if provided
        if prefix:
            prefixed_name = f"{prefix.upper()}_{key}"
            value = os.environ.get(prefixed_name)
            if value is not None:
                return value

        # If not found with provided prefix (or no prefix), try with default prefix
        prefixed_name = f"{self.with_prefix(key)}"
        value = os.environ.get(prefixed_name, default)
        if value is not None:
            return value

        # If not found with prefix (or no prefix), try without prefix
        value = os.environ.get(key, default)
        if value is not None:
            return value

        return default

    def get_environment(self) -> str:
        """Get the current environment name.

        Returns:
            The current environment name (e.g., 'development', 'test', 'production').
            Defaults to 'development' if not set.
        """
        return os.getenv("LOCCA_ENV", "development").lower()

    def is_development(self) -> bool:
        """Check if running in development environment.

        Returns:
            True if LOCCA_ENV is 'development' or not set (default).
        """
        return self.get_environment() in ("", "dev", "development")

    def is_testing(self) -> bool:
        """Check if running in test environment.

        Returns:
            True if LOCCA_ENV is 'test' or 'testing'.
        """
        return self.get_environment() in ("test", "testing")

    def is_production(self) -> bool:
        """Check if running in production environment.

        Returns:
            True if LOCCA_ENV is 'prod' or 'production'.
        """
        return self.get_environment() in ("prod", "production")

    def is_environment(self, env_name: str) -> bool:
        """Check if running in a specific environment.

        Args:
            env_name: The environment name to check against (case-insensitive).

        Returns:
            True if the current environment matches the given name.
        """
        if not isinstance(env_name, str):
            return False
        return self.get_environment() == env_name.lower()
