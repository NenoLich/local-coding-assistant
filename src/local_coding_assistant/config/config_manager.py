"""Configuration manager with 3-layer hierarchy support."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from local_coding_assistant.config.builder import ConfigBuilder
from local_coding_assistant.config.env_manager import EnvManager
from local_coding_assistant.config.field import ConfigFieldRegistry
from local_coding_assistant.config.path_manager import PathManager
from local_coding_assistant.config.schemas import AppConfig, ToolConfig
from local_coding_assistant.config.validation_engine import ValidationEngine
from local_coding_assistant.core.exceptions import ConfigError
from local_coding_assistant.core.protocols import IConfigManager
from local_coding_assistant.core.system_registry import (
    SystemCapabilityRegistry,
    system_capability_registry,
)
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("config.config_manager")


class ConfigManager(IConfigManager):
    """Configuration manager with hybrid field-based architecture.

    This manager uses a field-based approach instead of recursive dict processing,
    providing better performance, type safety, and maintainability.
    """

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
        self._path_manager = self.env_manager.path_manager
        self._system_registry = system_capability_registry

        # Resolve config paths
        self.config_paths = [
            self._path_manager.resolve_path(p) for p in (config_paths or [])
        ]

        # Default tool config paths resolved by tool_loader if none provided
        self.tool_config_paths = []
        if tool_config_paths is not None:
            self.tool_config_paths = [
                self._path_manager.resolve_path(p) for p in tool_config_paths
            ]

        # New architecture components
        self._field_registry = ConfigFieldRegistry()
        self._validation_engine = ValidationEngine(
            self._field_registry, self._system_registry
        )
        self._builder: ConfigBuilder | None = None

        # Configuration storage
        self._defaults_path = self._path_manager.resolve_path("@config/defaults.yaml")
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
            # Get the tools as a dictionary
            tools_dict = tool_loader.load_tool_configs()

            # Now assign it to the config (ensure _builder is not None)
            if self._builder:
                self._builder.update_tools(tools_dict)
            else:
                raise ConfigError("Global configuration is not initialized")

            logger.debug("Successfully loaded %d tools", len(tools_dict))
            return tools_dict

        except Exception as e:
            logger.error("Failed to load tools", error=str(e), exc_info=True)
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
            logger.error("Failed to reload tools", str(e), exc_info=True)
            raise ConfigError(f"Failed to reload tools: {e}") from e

    def load_global_config(self) -> AppConfig:
        """Load and initialize global configuration using ConfigField system.

        This method loads configuration from files and environment, creates the AppConfig
        instance, registers all ConfigFields, and initializes the configuration system.
        Validation is handled by the ConfigBuilder during the build process.

        Returns:
            Initialized AppConfig instance

        Raises:
            ConfigError: If configuration is invalid or validation fails
        """
        logger.info("Loading global configuration with ConfigField system")

        # 1. Load configuration data from files and environment
        try:
            config_data = self._load_config_data()
            logger.info("Configuration data loaded from files and environment")
        except Exception as e:
            error_msg = f"Failed to load configuration data: {e}"
            logger.error(error_msg)
            raise ConfigError(error_msg) from e

        # 2. Create AppConfig instance from loaded data - this auto-registers all ConfigFields
        try:
            config = AppConfig.from_dict(config_data)
            logger.info("AppConfig created and ConfigFields registered")
        except Exception as e:
            error_msg = f"Failed to create AppConfig: {e}"
            logger.error(error_msg)
            raise ConfigError(error_msg) from e

        # 3. Create builder for this configuration (builder handles validation internally)
        try:
            self._builder = ConfigBuilder(config, self._validation_engine)
            logger.debug("ConfigBuilder created")
        except Exception as e:
            error_msg = f"Failed to create ConfigBuilder: {e}"
            logger.error(error_msg)
            raise ConfigError(error_msg) from e

        # 4. Build final validated configuration (validation happens here)
        try:
            config = self._builder.build()
            logger.info("Global configuration loaded and validated successfully")
            return config
        except Exception as e:
            error_msg = f"Failed to build final configuration: {e}"
            logger.error(error_msg)
            raise ConfigError(error_msg) from e

    def _load_config_data(self) -> dict[str, Any]:
        """Load configuration data from all sources.

        Returns:
            Merged configuration dictionary

        Raises:
            ConfigError: If configuration loading fails
        """
        logger.info("Loading configuration from multiple sources")

        # Start with defaults
        config_data = self._load_defaults()

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

        return config_data

    def get_field_metadata(self, field_path: str) -> dict[str, Any]:
        """Get metadata for a specific configuration field.

        Args:
            field_path: The dot-notation path to the field

        Returns:
            Dictionary containing field metadata
        """
        field = self._field_registry.get_field(field_path)
        if not field:
            return {}

        return {
            "path": field.full_path,
            "parent_model": field.parent_model,
            "has_dependencies": field.has_dependencies,
            "dependencies": field.dependencies.model_dump()
            if field.dependencies is not None
            else None,
            "fallback_values": field.get_fallback_values(),
        }

    def validate_field(self, field_path: str, value: Any) -> dict[str, Any]:
        """Validate a single field value.

        Args:
            field_path: The dot-notation path to the field
            value: The value to validate

        Returns:
            Dictionary containing validation result
        """
        result = self._validation_engine.validate_field(field_path, value)
        return {
            "valid": result.valid,
            "error": result.error,
            "can_defer": result.can_defer,
            "missing_dependencies": result.missing_dependencies,
            "fallback_used": result.available_fallback,
        }

    def set_session_overrides(self, overrides: dict[str, Any]) -> None:
        """Set session-level configuration overrides.

        These overrides persist for the current session and will be applied
        on top of global config for all resolve() calls.

        Args:
            overrides: Dictionary of configuration overrides using dot notation
                      (e.g., {"llm.model_name": "gpt-4", "llm.temperature": 0.5})
        """
        if self._builder:
            self._builder.set_session_overrides(overrides)
        else:
            logger.error("Config builder not initialized")
            raise ConfigError("Config builder not initialized")

    @property
    def system_registry(self) -> SystemCapabilityRegistry:
        """Get the system capability registry.

        Returns:
            The SystemCapabilityRegistry instance
        """
        return self._system_registry

    def register_module(
        self, module_name: str, capabilities: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Register a module and update config with resolved pending validations.

        Args:
            module_name: Name of the module being registered
            capabilities: Dictionary of capabilities this module provides

        Returns:
            Dictionary of setting names to their new values (only actual changes)
        """
        # Get new resolutions from system registry
        new_resolved = self._system_registry.register_module(module_name, capabilities)

        # Compare with current values, only return actual changes
        actual_changes = {}
        if new_resolved:
            for setting_name, new_value in new_resolved:
                current_value = self._get_current_value(setting_name)
                if str(current_value).lower() != str(new_value).lower():
                    actual_changes[setting_name] = new_value

        # Apply only actual changes
        if actual_changes and self._builder:
            self._builder.set_session_overrides(actual_changes, skip_validation=True)
        # If no builder, the changes will be applied when config is loaded

        return actual_changes

    def register_capability(self, capabilities: list[str]) -> dict[str, Any]:
        """Register a capability directly.

        Args:
            capabilities: List of the capability to add

        Returns:
            Dictionary of setting names to their new values (only actual changes)
        """
        # Get new resolutions from system registry
        new_resolved = self._system_registry.register_capability(capabilities)

        # Compare with current values, only return actual changes
        actual_changes = {}
        if new_resolved:
            for setting_name, new_value in new_resolved:
                current_value = self._get_current_value(setting_name)
                if str(current_value).lower() != str(new_value).lower():
                    actual_changes[setting_name] = new_value

        # Apply only actual changes
        if actual_changes and self._builder:
            self._builder.set_session_overrides(actual_changes, skip_validation=True)
        # If no builder, the changes will be applied when config is loaded

        return actual_changes

    def unregister_capability(self, capabilities: list[str]) -> dict[str, Any]:
        """Unregister system capabilities and revalidate affected settings.

        This method removes the specified capabilities from the system and
        automatically revalidates any configuration settings that depend on
        those capabilities. It uses the ConfigBuilder to handle validation
        and fallback logic.

        Args:
            capabilities: List of capability names to unregister

        Returns:
            Dictionary of affected setting names to their current values
        """
        # Get affected setting names from system registry
        affected_settings = self._system_registry.unregister_capability(capabilities)

        # Get current values for affected settings before they change
        current_values = {}
        if affected_settings:
            for setting_name in affected_settings:
                current_value = self._get_current_value(setting_name)
                if current_value is not None:
                    current_values[setting_name] = current_value

        if affected_settings and self._builder:
            if current_values:
                logger.info(
                    "Revalidating settings after capability removal",
                    settings=current_values.keys(),
                    capabilities=capabilities,
                )

                # Use builder to handle validation and fallbacks
                # Mark as fallback handling to avoid corrupting pending validations
                self._builder.set_session_overrides(
                    current_values, pending_override=False
                )

        return current_values

    def _get_current_value(self, setting_name: str) -> Any:
        """Get the current value for a setting from the global config.

        Args:
            setting_name: Dot-notation path to the setting

        Returns:
            Current value or None if not found
        """
        try:
            # Navigate to the setting using dot notation
            parts = setting_name.split(".")
            current = self.global_config

            for part in parts:
                current = getattr(current, part)

            return current
        except (AttributeError, TypeError):
            logger.warning(f"Could not get current value for setting '{setting_name}'")
            return None

    def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status for debugging.

        Returns:
            Dictionary containing all system status information
        """
        return self._system_registry.get_system_status()

    def clear_session_overrides(self) -> None:
        """Clear all session-level configuration overrides."""
        if self._builder:
            logger.info("Clearing all session overrides")
            self._builder.clear_session_overrides()

    def resolve(self) -> AppConfig:
        """Resolve configuration with all layers applied.

        Layer priority (highest to lowest):
        1. Call overrides (highest priority)
        2. Session overrides
        3. Global config (lowest priority)

        Returns:
            AppConfig: The resolved and validated configuration

        Raises:
            ConfigError: If no builder is available
        """
        if self._builder:
            # Build and return the final configuration
            return self._builder.build()
        else:
            logger.error("Config builder not initialized")
            raise ConfigError("Config builder not initialized")

    def get_cache_info(self) -> dict:
        """Get cache statistics for the validation engine.

        Returns:
            dict: Dictionary containing cache statistics
        """
        validation_engine_initialized = self._validation_engine is not None
        return {
            "validation_engine_initialized": validation_engine_initialized,
            "cache_stats": self._validation_engine.get_cache_stats()
            if validation_engine_initialized
            else None,
        }

    def get_config(
        self,
        provider: str | None = None,
        model_name: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> AppConfig:
        """Get configuration with all layers applied.
        DEPRECATED: Use call_overrides and then global_config property

        This is the main public method that maintains backward compatibility.

        Args:
            provider: Optional provider override
            model_name: Optional model name override
            overrides: Optional additional overrides

        Returns:
            AppConfig: The resolved configuration

        Raises:
            ConfigError: If configuration is not loaded or resolution fails
        """
        overrides = overrides or {}
        if self._builder:
            provider_override = {"llm.provider": provider} if provider else {}
            model_name_override = {"llm.model_name": model_name} if model_name else {}
            overrides.update(provider_override)
            overrides.update(model_name_override)

            return self._builder.set_call_overrides(overrides)
        else:
            logger.error("Config builder not initialized")
            raise ConfigError("Config builder not initialized")

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
        resolved_path = self._path_manager.resolve_path(path)
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
    def global_config(self) -> AppConfig:
        """Get the current global configuration with automatic session override resolution.

        Returns:
            A wrapped AppConfig that automatically resolves session overrides when accessed.

        Raises:
            ConfigError: If no configuration is loaded.
        """
        return self.resolve()

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

        # Ensure global config is loaded
        if self._builder is None:
            raise ConfigError(
                "No configuration loaded. Call load_global_config() first."
            )

        try:
            # Resolve path using PathManager and ensure parent directory exists
            resolved_path = self._path_manager.resolve_path(path, ensure_parent=True)

            # Get the current config
            config_data = self.global_config.model_dump(
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
        if self._builder:
            # Build and return the final configuration
            return self._builder.get_session_overrides()
        else:
            logger.error("Config builder not initialized")
            raise ConfigError("Config builder not initialized")

    @property
    def path_manager(self) -> PathManager:
        return self._path_manager
