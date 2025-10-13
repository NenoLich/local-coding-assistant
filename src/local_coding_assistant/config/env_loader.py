"""Environment variable loading utilities for Local Coding Assistant."""

import os
from pathlib import Path
from typing import Any

from local_coding_assistant.utils.logging import get_logger

logger = get_logger("config.env_loader")


class EnvLoader:
    """Loads and parses environment variables for configuration."""

    def __init__(self, env_prefix: str = "LOCCA_", env_paths: list[Path] | None = None):
        """Initialize the environment loader.

        Args:
            env_prefix: Prefix for environment variables to load (default: "LOCCA_")
            env_paths: Optional list of .env file paths to load (default: auto-detect)
        """
        self.env_prefix = env_prefix
        self.env_paths = env_paths or self._get_default_env_paths()

    def _get_default_env_paths(self) -> list[Path]:
        """Get default .env file paths to check."""
        # Try to find project root by looking for pyproject.toml
        current_path = Path(__file__).parent
        for parent in [current_path, *list(current_path.parents)]:
            if (parent / "pyproject.toml").exists():
                return [parent / ".env", parent / ".env.local"]
        # Fallback to current working directory
        return [Path.cwd() / ".env", Path.cwd() / ".env.local"]

    def load_env_files(self) -> None:
        """Load .env files if python-dotenv is available.

        Raises:
            ConfigError: If .env file exists but cannot be loaded
        """
        try:
            from dotenv import load_dotenv
        except ImportError:
            logger.debug("python-dotenv not installed, skipping .env file loading")
            return

        loaded_any = False
        for env_path in self.env_paths:
            if env_path.exists():
                try:
                    load_dotenv(
                        env_path, override=loaded_any
                    )  # Later files override earlier ones
                    logger.info(f"Loaded environment variables from {env_path}")
                    loaded_any = True
                except Exception as e:
                    from local_coding_assistant.core.exceptions import ConfigError

                    raise ConfigError(
                        f"Failed to load .env file {env_path}: {e}"
                    ) from e

        if not loaded_any:
            logger.debug("No .env files found to load")

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
