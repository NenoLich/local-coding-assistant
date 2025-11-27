"""Configuration manager with 3-layer hierarchy support."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from local_coding_assistant.config.env_manager import EnvManager
from local_coding_assistant.config.schemas import AppConfig, ToolConfig
from local_coding_assistant.core.exceptions import ConfigError
from local_coding_assistant.core.protocols import IConfigManager
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("config.config_manager")


class ConfigManager(IConfigManager):
    """Configuration manager with 3-layer hierarchy support."""

    def __init__(
        self,
        config_paths: list[Path | str] | None = None,
        env_manager: EnvManager | None = None,
        tool_config_paths: list[Path | str] | None = None,
    ):
        """Initialize the config manager.

        Args:
            config_paths: Optional list of paths to main YAML config files to load
            env_manager: Optional EnvManager instance (creates default if not provided)
            tool_config_paths: Optional list of paths to tool YAML config files
        """
        self.env_manager: EnvManager = env_manager or EnvManager()
        self.path_manager = self.env_manager.path_manager

        # Resolve config paths
        self.config_paths = [
            self.path_manager.resolve_path(p) for p in (config_paths or [])
        ]

        # Default tool config paths if none provided
        if tool_config_paths is None:
            self.tool_config_paths = [
                self.path_manager.resolve_path("@config/tools.default.yaml"),
                self.path_manager.resolve_path("@config/tools.local.yaml"),
            ]
        else:
            self.tool_config_paths = [
                self.path_manager.resolve_path(p) for p in tool_config_paths
            ]

        # 3-layer configuration storage
        self._defaults_path = self.path_manager.resolve_path("@config/defaults.yaml")
        self._global_config: AppConfig | None = None
        self._session_overrides: dict[str, Any] = {}
        self._loaded_tools: dict[str, ToolConfig] = {}

    def _load_tools(self) -> dict[str, ToolConfig]:
        """Internal method to load tools using ToolLoader.

        Returns:
            Dictionary of loaded tools

        Raises:
            ConfigError: If there's an error loading tools
        """
        try:
            from local_coding_assistant.config.tool_loader import ToolLoader

            tool_loader = ToolLoader(
                env_manager=self.env_manager, tool_config_paths=self.tool_config_paths
            )
            tools = tool_loader.load_tool_configs()
            logger.debug("Successfully loaded %d tools", len(tools))
            return tools

        except Exception as e:
            logger.error("Failed to load tools: %s", str(e), exc_info=True)
            raise ConfigError(f"Failed to load tools: {e}") from e

    def get_tools(self) -> dict[str, ToolConfig]:
        """Get all configured tools.

        Implements IConfigManager.get_tools().

        Returns:
            Dictionary mapping tool names to their configuration.
        """
        if not self._loaded_tools:  # Check if _loaded_tools is empty
            self._loaded_tools = self._load_tools()
        tools_config = self._loaded_tools
        if not isinstance(tools_config, dict):
            logger.warning("Invalid tools configuration - expected a dictionary")
            return {}
        return tools_config

    def reload_tools(self) -> None:
        """Reload tools configuration from all sources.

        Implements IConfigManager.reload_tools().
        """
        logger.info("Reloading tools configuration...")
        try:
            # Clear any cached tools
            if hasattr(self, "_loaded_tools"):
                del self._loaded_tools

            # Reload tools
            self._loaded_tools = self._load_tools()
            logger.info("Successfully reloaded %d tools", len(self._loaded_tools))
        except Exception as e:
            logger.error("Failed to reload tools: %s", str(e), exc_info=True)
            raise ConfigError(f"Failed to reload tools: {e}") from e

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

        # Load tool configurations
        try:
            tool_configs = self.get_tools()
            if tool_configs:
                config_data["tools"] = {"tools": list(tool_configs.values())}
        except Exception as e:
            logger.error(f"Failed to load tool configurations: {e}")
            raise ConfigError(f"Failed to load tool configurations: {e}") from e

        # Merge YAML files (in order provided)
        for config_path in self.config_paths:
            yaml_data = self._load_yaml_file(config_path)

            if yaml_data is not None:
                config_data = self._deep_merge(config_data, yaml_data)
                logger.debug(f"Merged YAML config from {config_path}")
            else:
                logger.warning(
                    f"YAML file {config_path} did not load as dict, skipping"
                )

        # Merge environment variables (highest priority in global layer)
        env_data = self.env_manager.get_config_from_env()
        if env_data:
            config_data = self._deep_merge(config_data, env_data)

        # Create and validate final config
        try:
            config = AppConfig.from_dict(config_data)
            logger.info("Global configuration loaded and validated successfully")
            self._global_config = config
            return config
        except ValidationError as e:
            error_msg = f"Invalid global configuration: {e}"
            logger.error(error_msg)
            raise ConfigError(error_msg) from e

    def set_session_overrides(self, overrides: dict[str, Any]) -> None:
        """Set session-level configuration overrides.

        These overrides persist for the current session and will be applied
        on top of global config for all resolve() calls.

        Args:
            overrides: Dictionary of configuration overrides using dot notation
                      (e.g., {"llm.model_name": "gpt-4", "llm.temperature": 0.5})

        Raises:
            LLMError: If the overrides contain invalid LLM configuration values
        """
        logger.info(f"Setting session overrides: {list(overrides.keys())}")

        # Validate overrides by attempting to resolve configuration with them
        if overrides:
            try:
                # Apply overrides to global config (similar to resolve method)
                if self._global_config is not None:
                    resolved_data = self._global_config.to_dict()

                    # Apply session overrides
                    session_overrides = self._apply_overrides_to_dict(
                        resolved_data, overrides
                    )
                    resolved_data = self._deep_merge(resolved_data, session_overrides)

                    # Try to create AppConfig to validate the merged configuration
                    AppConfig.from_dict(resolved_data)
                else:
                    # If no global config, just validate the overrides directly
                    AppConfig.from_dict(overrides)

            except ValidationError as e:
                error_msg = f"Invalid resolved configuration: {e}"
                logger.error(error_msg)

                # Check if this is an LLM configuration validation error
                error_details = str(e)
                if any(
                    field in error_details.lower()
                    for field in ["temperature", "max_tokens", "llm"]
                ):
                    from local_coding_assistant.core.exceptions import LLMError

                    raise LLMError(
                        f"Configuration update validation failed: {e}"
                    ) from e
                else:
                    raise ConfigError(error_msg) from e

        # Only set overrides if validation passed (or if no overrides provided)
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
                raise ConfigError(error_msg) from e

    def _load_config_file(self, path: Path | str) -> dict[str, Any]:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML file (can be relative or use @ prefixes)

        Returns:
            Dictionary with the loaded configuration

        Raises:
            ConfigError: If the file cannot be loaded or parsed
        """
        # Resolve path using PathManager
        resolved_path = self.path_manager.resolve_path(path)
        try:
            with open(resolved_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)
                if config is None:
                    logger.warning(f"Config file {resolved_path} is empty")
                    return {}
                if not isinstance(config, dict):
                    logger.warning(f"Config file {resolved_path} is not a dictionary")
                    return {}
                return config
        except (yaml.YAMLError, OSError) as e:
            raise ConfigError(
                f"Failed to load config from {path} (resolved to: {resolved_path}): {e}"
            ) from e

    def _load_defaults(self) -> dict[str, Any]:
        """Load default configuration values."""
        if self._defaults_path.exists():
            config_data = self._load_config_file(self._defaults_path)
            logger.debug(f"Loaded default config: {len(config_data)} keys")
            return config_data
        else:
            logger.debug("No defaults file found, using empty defaults")
            return {}

    def _load_yaml_file(self, file_path: Path) -> dict[str, Any] | None:
        """Load YAML file and return as dictionary.

        Args:
            file_path: Path to the YAML file (can be relative or use @ prefixes)

        Returns:
            Configuration dictionary or None if file doesn't exist

        Raises:
            ConfigError: If file exists but is invalid YAML
        """
        logger.debug(f"Loading YAML file: {file_path}")

        try:
            return self._load_config_file(file_path)
        except ConfigError as e:
            if "No such file or directory" in str(e):
                logger.debug(f"YAML file {file_path} does not exist, skipping")
                return None
            raise  # Re-raise other ConfigError exceptions

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

    def save_config(self, path: Path | str | None = None) -> None:
        """Save the current configuration to a file.

        Args:
            path: Path to save the config to (can be relative or use @ prefixes).
                  If None, saves to the first config path.

        Raises:
            ConfigError: If no config paths are configured or save fails
        """
        if path is None:
            if not self.config_paths:
                raise ConfigError("No config paths configured to save to")
            path = self.config_paths[0]

        try:
            # Resolve path using PathManager and ensure parent directory exists
            resolved_path = self.path_manager.resolve_path(path, ensure_parent=True)

            # Ensure global config is loaded
            if self._global_config is None:
                raise ConfigError(
                    "No configuration loaded. Call load_global_config() first."
                )

            # Get the current config
            config_data = self._global_config.model_dump(
                exclude_unset=True, exclude_defaults=True, exclude_none=True
            )

            # Save to file
            with open(resolved_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config_data, f, sort_keys=False)

            logger.info("Configuration saved to %s", resolved_path)

        except (OSError, yaml.YAMLError) as e:
            error_msg = f"Failed to save configuration to {path}: {e}"
            logger.error(error_msg)
            raise ConfigError(error_msg) from e

    @property
    def session_overrides(self) -> dict[str, Any]:
        """Get the current session overrides."""
        return deepcopy(self._session_overrides)
