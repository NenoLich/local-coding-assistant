"""Configuration manager with 3-layer hierarchy support."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from local_coding_assistant.config.env_loader import EnvLoader
from local_coding_assistant.config.schemas import AppConfig
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("config.config_manager")


class ConfigManager:
    """Configuration manager with 3-layer hierarchy support.

    Provides a 3-layer configuration system:
    1. Global layer: Base configuration from env/YAML/defaults
    2. Session layer: Runtime-modifiable session overrides
    3. Call layer: Temporary overrides for individual function calls

    Priority order (highest to lowest): Call → Session → Global
    """

    def __init__(
        self,
        config_paths: list[Path] | None = None,
        env_loader: EnvLoader | None = None,
    ):
        """Initialize the config manager.

        Args:
            config_paths: Optional list of paths to YAML config files to load
            env_loader: Optional EnvLoader instance (creates default if not provided)
        """
        self.config_paths = config_paths or []
        self.env_loader = env_loader or EnvLoader()
        self._defaults_path = Path(__file__).parent / "defaults.yaml"

        # 3-layer configuration storage
        self._global_config: AppConfig | None = None
        self._session_overrides: dict[str, Any] = {}

    def load_global_config(self) -> AppConfig:
        """Load and merge configuration from all sources into global layer.

        Environment variables (including those loaded from .env files) are
        automatically available and have the highest priority in the global layer.

        Returns:
            Merged AppConfig instance

        Raises:
            ConfigError: If configuration is invalid or files don't exist
        """
        logger.info("Loading global configuration from multiple sources")

        # Start with defaults
        config_data = self._load_defaults()
        logger.debug(f"Loaded defaults: {len(config_data)} keys")

        # Merge YAML files (in order provided)
        for config_path in self.config_paths:
            yaml_data = self._load_yaml_file(config_path)
            config_data = self._deep_merge(config_data, yaml_data)
            logger.debug(f"Merged YAML config from {config_path}")

        # Merge environment variables (highest priority in global layer)
        env_data = self.env_loader.get_config_from_env()
        if env_data:
            config_data = self._deep_merge(config_data, env_data)
            logger.debug(f"Merged {len(env_data)} environment variables")

        # Create and validate final config
        try:
            config = AppConfig.from_dict(config_data)
            logger.info("Global configuration loaded and validated successfully")
            self._global_config = config
            return config
        except ValidationError as e:
            error_msg = f"Invalid global configuration: {e}"
            logger.error(error_msg)
            from local_coding_assistant.core.exceptions import ConfigError

            raise ConfigError(error_msg) from e

    def set_session_overrides(self, overrides: dict[str, Any]) -> None:
        """Set session-level configuration overrides.

        These overrides persist for the current session and will be applied
        on top of global config for all resolve() calls.

        Args:
            overrides: Dictionary of configuration overrides using dot notation
                      (e.g., {"llm.model_name": "gpt-4", "llm.temperature": 0.5})
        """
        logger.info(f"Setting session overrides: {list(overrides.keys())}")
        self._session_overrides = deepcopy(overrides)

    def clear_session_overrides(self) -> None:
        """Clear all session-level configuration overrides."""
        logger.info("Clearing all session overrides")
        self._session_overrides.clear()

    def resolve(
        self,
        provider: str | None = None,
        model_name: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> AppConfig:
        """Resolve configuration with all 3 layers applied.

        Layer priority (highest to lowest):
        1. Call overrides (function parameters)
        2. Session overrides (runtime session settings)
        3. Global defaults (persistent base config)

        Args:
            provider: Optional provider override for this call
            model_name: Optional model_name override for this call
            overrides: Optional additional overrides for this call

        Returns:
            Resolved AppConfig with all layers merged

        Raises:
            ConfigError: If no global config is loaded or resolution fails
        """
        if self._global_config is None:
            from local_coding_assistant.core.exceptions import ConfigError

            raise ConfigError(
                "Global configuration not loaded. Call load_global_config() first."
            )

        # Start with global config
        resolved_data = self._global_config.to_dict()

        # Apply session overrides
        if self._session_overrides:
            session_overrides = self._apply_overrides_to_dict(
                resolved_data, self._session_overrides
            )
            resolved_data = self._deep_merge(resolved_data, session_overrides)
            logger.debug(f"Applied {len(self._session_overrides)} session overrides")

        # Apply call overrides
        call_overrides = {}
        if provider is not None or model_name is not None or overrides:
            if provider is not None:
                call_overrides["llm.provider"] = provider
            if model_name is not None:
                call_overrides["llm.model_name"] = model_name
            if overrides:
                call_overrides.update(overrides)

            if call_overrides:
                call_overrides = self._apply_overrides_to_dict(
                    resolved_data, call_overrides
                )
                resolved_data = self._deep_merge(resolved_data, call_overrides)
                logger.debug(f"Applied {len(call_overrides)} call overrides")

        # Create final config
        try:
            return AppConfig.from_dict(resolved_data)
        except ValidationError as e:
            error_msg = f"Invalid resolved configuration: {e}"
            logger.error(error_msg)

            # Check if this is an LLM configuration validation error
            # If so, raise LLMError instead of ConfigError since it's a llm-level concern
            error_details = str(e)
            if any(
                field in error_details.lower()
                for field in ["temperature", "max_tokens", "llm"]
            ):
                from local_coding_assistant.core.exceptions import LLMError

                raise LLMError(f"Configuration update validation failed: {e}") from e
            else:
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

    def _load_yaml_file(self, file_path: Path) -> dict[str, Any] | None:
        """Load YAML file and return as dictionary.

        Args:
            file_path: Path to the YAML file

        Returns:
            Configuration dictionary or None if file doesn't exist

        Raises:
            ConfigError: If file exists but is invalid YAML
        """
        logger.debug(f"Loading YAML file: {file_path}")
        logger.debug(f"File exists: {file_path.exists()}")

        if not file_path.exists():
            logger.debug(f"YAML file {file_path} does not exist, skipping")
            return None

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

    def _apply_overrides_to_dict(
        self, base_dict: dict[str, Any], overrides: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply dot-notation overrides to a nested dictionary.

        Args:
            base_dict: Base dictionary to apply overrides to
            overrides: Dictionary of overrides using dot notation keys

        Returns:
            Dictionary with overrides applied
        """
        result = deepcopy(base_dict)

        for key, value in overrides.items():
            if "." not in key:
                # Simple top-level key
                result[key] = value
            else:
                # Dot-notation nested key
                key_parts = key.split(".")
                current = result

                # Navigate/create nested structure
                for part in key_parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]

                # Set the final value
                current[key_parts[-1]] = value

        return result

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
        result = deepcopy(base)

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

    @property
    def global_config(self) -> AppConfig | None:
        """Get the current global configuration."""
        return self._global_config

    @property
    def session_overrides(self) -> dict[str, Any]:
        """Get the current session overrides."""
        return deepcopy(self._session_overrides)


# Global config manager instance for convenience
_config_manager = None


def get_config_manager(config_paths: list[Path] | None = None) -> ConfigManager:
    """Get or create a global config manager instance.

    Args:
        config_paths: Optional list of config file paths

    Returns:
        ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_paths)
    return _config_manager


def load_config(config_paths: list[Path] | None = None) -> AppConfig:
    """Load application configuration.

    Args:
        config_paths: Optional list of config file paths

    Returns:
        Loaded and merged AppConfig
    """
    manager = ConfigManager(config_paths)
    return manager.load_global_config()
