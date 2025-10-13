"""Configuration loader that merges ENV → YAML → Defaults."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from local_coding_assistant.config.env_loader import EnvLoader
from local_coding_assistant.config.schemas import AppConfig
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("config.loader")


class ConfigLoader:
    """Loads and merges configuration from multiple sources.

    Priority order (highest to lowest):
    1. Environment variables (including .env files loaded by bootstrap)
    2. YAML configuration files
    3. Default values

    Environment variables are loaded automatically by the bootstrap process
    from .env and .env.local files in the project root before configuration
    loading begins.
    """

    def __init__(
        self,
        config_paths: list[Path] | None = None,
        env_loader: EnvLoader | None = None,
    ):
        """Initialize the config loader.

        Args:
            config_paths: Optional list of paths to YAML config files to load
            env_loader: Optional EnvLoader instance (creates default if not provided)
        """
        self.config_paths = config_paths or []
        self.env_loader = env_loader or EnvLoader()
        self._defaults_path = Path(__file__).parent / "defaults.yaml"

    def load_config(self) -> AppConfig:
        """Load and merge configuration from all sources.

        Environment variables (including those loaded from .env files) are
        automatically available and have the highest priority.

        Returns:
            Merged AppConfig instance

        Raises:
            ConfigError: If configuration is invalid or files don't exist
        """
        logger.info("Loading configuration from multiple sources")
        # Start with defaults
        config_data = self._load_defaults()
        logger.debug(f"Loaded defaults: {len(config_data)} keys")

        # Merge YAML files (in order provided)
        for config_path in self.config_paths:
            yaml_data = self._load_yaml_file(config_path)
            config_data = self._deep_merge(config_data, yaml_data)
            logger.debug(f"Merged YAML config from {config_path}")

        # Merge environment variables (highest priority)
        env_data = self.env_loader.get_config_from_env()
        if env_data:
            config_data = self._deep_merge(config_data, env_data)
            logger.debug(f"Merged {len(env_data)} environment variables")

        # Create and validate final config
        try:
            config = AppConfig.from_dict(config_data)
            logger.info("Configuration loaded and validated successfully")
            return config
        except ValidationError as e:
            error_msg = f"Invalid configuration: {e}"
            logger.error(error_msg)
            from local_coding_assistant.core.exceptions import ConfigError

            raise ConfigError(error_msg) from e

    def _load_defaults(self) -> dict[str, Any]:
        """Load default configuration values."""
        if self._defaults_path.exists():
            config_data = self._load_yaml_file(self._defaults_path)
            # Handle case where _load_yaml_file returns None
            if config_data is None:
                logger.warning("_load_yaml_file returned None, using empty dict")
                config_data = {}
            logger.debug(f"Loaded default config: {len(config_data)} keys")
            return config_data
        else:
            logger.debug("No defaults file found, using empty defaults")
            return {}

    def _load_yaml_file(self, file_path: Path) -> dict[str, Any]:
        """Load YAML file and return as dictionary.

        Args:
            file_path: Path to the YAML file

        Returns:
            Configuration dictionary

        Raises:
            ConfigError: If file doesn't exist or is invalid YAML
        """
        logger.debug(f"Loading YAML file: {file_path}")
        logger.debug(f"File exists: {file_path.exists()}")

        if not file_path.exists():
            error_msg = f"Config file not found: {file_path}"
            logger.error(error_msg)
            from local_coding_assistant.core.exceptions import ConfigError

            raise ConfigError(error_msg)

        try:
            with open(file_path, encoding="utf-8") as file:
                raw_content = file.read()
                logger.debug(f"File content length: {len(raw_content)} characters")
                # Ensure we always return a dict, never None
                config_data = yaml.safe_load(raw_content)
                if config_data is None:
                    logger.warning(
                        f"YAML file {file_path} loaded as None, using empty dict"
                    )
                    config_data = {}
                elif not isinstance(config_data, dict):
                    logger.warning(
                        f"YAML file {file_path} did not load as dict, got {type(config_data)}, using empty dict"
                    )
                    config_data = {}

                logger.debug(
                    f"Loaded YAML config from {file_path}: {len(config_data)} keys"
                )
                return config_data
        except UnicodeDecodeError as e:
            error_msg = f"Unicode decode error in {file_path}: {e}. File may not be UTF-8 encoded."
            logger.error(error_msg)
            from local_coding_assistant.core.exceptions import ConfigError

            raise ConfigError(error_msg) from e
        except yaml.YAMLError as e:
            error_msg = f"Invalid YAML in {file_path}: {e}"
            logger.error(error_msg)
            from local_coding_assistant.core.exceptions import ConfigError

            raise ConfigError(error_msg) from e

    def _set_nested_value(
        self, data: dict[str, Any], key_parts: list[str], value: str
    ) -> None:
        """Set a nested dictionary value from flattened key parts.

        Args:
            data: Dictionary to modify
            key_parts: List of key parts (e.g., ['llm', 'model_name'])
            value: Value to set
        """
        current = data

        # Navigate/create nested structure
        for _i, part in enumerate(key_parts[:-1]):
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set the final value (convert types as needed)
        final_key = key_parts[-1]
        current[final_key] = self._convert_env_value(value)

    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type.

        Args:
            value: String value from environment

        Returns:
            Converted value (bool, int, float, or string)
        """
        # Boolean conversion
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # Integer conversion
        try:
            return int(value)
        except ValueError:
            pass

        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string if no conversion possible
        return value

    def _deep_merge(
        self, base: dict[str, Any], overlay: dict[str, Any]
    ) -> dict[str, Any]:
        """Deep merge two dictionaries.

        Args:
            base: Base dictionary
            overlay: Dictionary to merge on top

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in overlay.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result


# Global loader instance for convenience
_config_loader = None


def get_config_loader(config_paths: list[Path] | None = None) -> ConfigLoader:
    """Get or create a global config loader instance.

    Args:
        config_paths: Optional list of config file paths

    Returns:
        ConfigLoader instance
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_paths)
    return _config_loader


def load_config(config_paths: list[Path] | None = None) -> AppConfig:
    """Load application configuration.

    Args:
        config_paths: Optional list of config file paths

    Returns:
        Loaded and merged AppConfig
    """
    loader = ConfigLoader(config_paths)
    return loader.load_config()
