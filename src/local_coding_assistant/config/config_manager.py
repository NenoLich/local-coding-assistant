"""Configuration manager with 3-layer hierarchy support."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from copy import deepcopy
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import ValidationError

from local_coding_assistant.config.env_manager import EnvManager
from local_coding_assistant.config.path_manager import PathManager
from local_coding_assistant.config.schemas import AppConfig, ToolConfig, ToolConfigList
from local_coding_assistant.core.exceptions import ConfigError
from local_coding_assistant.core.protocols import IConfigManager
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("config.config_manager")

T = TypeVar("T")


def dict_cache(maxsize: int = 32):
    """Cache decorator that can handle dictionary arguments by converting them to a stable key.

    Args:
        maxsize: Maximum number of entries to keep in the cache
    """

    def decorator(func: Callable) -> Callable:
        cache: dict[str, Any] = {}
        hits = 0
        misses = 0

        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal hits, misses

            # Skip caching if the first argument is 'self'
            instance = args[0] if args and hasattr(args[0], "__class__") else None
            method_args = args[1:] if instance is not None else args

            # Create a stable key from the arguments
            cache_key = _generate_cache_key(method_args, kwargs)

            # Check cache
            if cache_key in cache:
                hits += 1
                return cache[cache_key]

            # Cache miss - call the function
            misses += 1
            result = func(*args, **kwargs)

            # Cache the result if we're under the limit
            if len(cache) < maxsize:
                cache[cache_key] = result

            return result

        def cache_clear() -> None:
            """Clear the cache and reset statistics."""
            nonlocal hits, misses
            cache.clear()
            hits = 0
            misses = 0

        def cache_info() -> dict:
            """Get cache statistics.

            Returns:
                dict: Dictionary containing cache statistics including:
                    - 'hits': Number of cache hits
                    - 'misses': Number of cache misses
                    - 'maxsize': Maximum cache size
                    - 'currsize': Current cache size
            """
            return {
                "hits": hits,
                "misses": misses,
                "maxsize": maxsize,
                "currsize": len(cache),
            }

        # Attach cache management methods
        wrapper.cache_clear = cache_clear  # type: ignore
        wrapper.cache_info = cache_info  # type: ignore
        wrapper.cache = cache  # type: ignore  # For debugging/inspection

        return wrapper

    return decorator


def _generate_cache_key(args: tuple, kwargs: dict) -> str:
    """Generate a stable cache key from function arguments."""

    def _to_hashable(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        elif isinstance(value, (list, tuple)):
            return tuple(_to_hashable(v) for v in value)
        elif isinstance(value, dict):
            return tuple(sorted((k, _to_hashable(v)) for k, v in value.items()))
        elif hasattr(value, "__dict__"):
            return _to_hashable(vars(value))
        return str(value)  # Fallback for other types

    # Convert all arguments to hashable types
    key_parts = (_to_hashable(args), _to_hashable(kwargs))

    # Create a stable string representation and hash it
    key_str = json.dumps(key_parts, sort_keys=True)
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()


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
        self._path_manager = self.env_manager.path_manager

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

        # 3-layer configuration storage
        self._defaults_path = self._path_manager.resolve_path("@config/defaults.yaml")
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
            # Get the tools as a dictionary
            tools_dict = tool_loader.load_tool_configs()

            # Convert the dictionary to a ToolConfigList
            tool_config_list = ToolConfigList(tools=list(tools_dict.values()))

            # Now assign it to the config (ensure _global_config is not None)
            if self._global_config is not None:
                self._global_config.tools = tool_config_list
            else:
                raise ConfigError("Global configuration is not initialized")

            self._resolve_dicts.cache_clear()

            logger.debug("Successfully loaded %d tools", len(tools_dict))
            return tools_dict

        except Exception as e:
            logger.error("Failed to load tools", str(e), exc_info=True)
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
            self._resolve_dicts.cache_clear()

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
        if not overrides:
            return

        logger.info("Setting session overrides", overrides=overrides)

        # Create a copy of current overrides and update with new ones
        new_overrides = deepcopy(self._session_overrides or {})
        new_overrides.update(overrides)

        try:
            # Apply overrides to global config (similar to resolve method)
            if self._global_config is not None:
                resolved_data = self._global_config.to_dict()

                # Apply session overrides
                session_overrides = self._apply_overrides_to_dict(
                    resolved_data, new_overrides
                )
                resolved_data = self._deep_merge(resolved_data, session_overrides)

                # Try to create AppConfig to validate the merged configuration
                AppConfig.from_dict(resolved_data)
            else:
                # If no global config, just validate the overrides directly
                AppConfig.from_dict(new_overrides)

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

                raise LLMError(f"Configuration update validation failed: {e}") from e
            else:
                raise ConfigError(error_msg) from e

        # Only set overrides if validation passed (or if no overrides provided)
        self._session_overrides = new_overrides
        self._resolve_dicts.cache_clear()

    def clear_session_overrides(self) -> None:
        """Clear all session-level configuration overrides."""
        if self._session_overrides:
            logger.info("Clearing all session overrides")
            self._session_overrides.clear()
            self._resolve_dicts.cache_clear()

    @dict_cache(maxsize=32)
    def _resolve_dicts(
        self,
        global_config: dict,
        session_overrides: dict,
        call_overrides: dict,
    ) -> AppConfig:
        """Internal method that resolves configuration dictionaries into an AppConfig.

        This method is cached based on the input dictionaries.

        Args:
            global_config: Base configuration dictionary
            session_overrides: Session-level overrides
            call_overrides: Call-specific overrides (highest priority)

        Returns:
            AppConfig: The resolved and validated configuration

        Raises:
            ConfigError: If configuration resolution or validation fails
            LLMError: If LLM-specific validation fails
        """
        # Start with global config
        resolved_data = deepcopy(global_config)

        # Apply session overrides
        if session_overrides:
            session_overrides = self._apply_overrides_to_dict(
                resolved_data, session_overrides
            )
            resolved_data = self._deep_merge(resolved_data, session_overrides)

        # Apply call overrides
        if call_overrides:
            call_overrides = self._apply_overrides_to_dict(
                resolved_data, call_overrides
            )
            resolved_data = self._deep_merge(resolved_data, call_overrides)

        # Convert to AppConfig and validate
        try:
            return AppConfig.from_dict(resolved_data)
        except ValidationError as e:
            error_msg = f"Invalid resolved configuration: {e}"
            logger.error(error_msg)

            # Handle LLM-specific validation errors
            error_details = str(e).lower()
            if any(
                field in error_details for field in ["temperature", "max_tokens", "llm"]
            ):
                from local_coding_assistant.core.exceptions import LLMError

                raise LLMError(f"Configuration validation failed: {e}") from e
            raise ConfigError(error_msg) from e

    def resolve(
        self,
        global_config: dict | None = None,
        session_overrides: dict | None = None,
        call_overrides: dict | None = None,
    ) -> AppConfig:
        """Resolve configuration with all layers applied.

        Layer priority (highest to lowest):
        1. Call overrides (highest priority)
        2. Session overrides
        3. Global config (lowest priority)

        Args:
            global_config: Base configuration dictionary. If None, uses instance's global config.
            session_overrides: Session-level overrides. If None, uses instance's session overrides.
            call_overrides: Call-specific overrides (highest priority). If None, uses empty dict.

        Returns:
            AppConfig: The resolved and validated configuration

        Raises:
            ConfigError: If no global config is available or resolution fails
            LLMError: If LLM-specific validation fails
        """
        # Use instance values if not provided
        if global_config is None:
            if self._global_config is None:
                raise ConfigError("No global config available")
            global_config = self._global_config.to_dict()

        session_overrides = (
            session_overrides
            if session_overrides is not None
            else self._session_overrides
        )
        call_overrides = call_overrides if call_overrides is not None else {}

        return self._resolve_dicts(global_config, session_overrides, call_overrides)

    def get_cache_info(self) -> dict:
        """Get cache statistics for the _resolve_dicts method.

        Returns:
            dict: Dictionary containing cache statistics including:
                - 'hits': Number of cache hits
                - 'misses': Number of cache misses
                - 'maxsize': Maximum cache size
                - 'currsize': Current cache size
        """
        if not hasattr(self._resolve_dicts, "cache_info"):
            return {
                "hits": 0,
                "misses": 0,
                "maxsize": 0,
                "currsize": 0,
                "enabled": False,
            }
        return self._resolve_dicts.cache_info()

    def get_config(
        self,
        provider: str | None = None,
        model_name: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> AppConfig:
        """Get configuration with all layers applied.

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
        if self._global_config is None:
            raise ConfigError(
                "Global configuration not loaded. Call load_global_config() first."
            )

        # Prepare call overrides
        call_overrides = {}
        if provider is not None:
            call_overrides["llm.provider"] = provider
        if model_name is not None:
            call_overrides["llm.model_name"] = model_name
        if overrides:
            call_overrides.update(overrides)

        # Apply all layers
        resolved_dict = self.resolve(
            global_config=self._global_config.to_dict(),
            session_overrides=self._session_overrides,
            call_overrides=call_overrides,
        )

        return resolved_dict

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
        if not isinstance(base_dict, dict):
            base_dict = {}

        result = deepcopy(base_dict)

        for key_path, value in overrides.items():
            if not key_path:
                continue

            parts = key_path.split(".")
            current = result

            # Navigate to the parent of the target key
            for part in parts[:-1]:
                if not isinstance(current, dict):
                    current = {}
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Set the final value
            if parts:
                current[parts[-1]] = value

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
    def global_config(self) -> AppConfig:
        """Get the current global configuration with automatic session override resolution.

        Returns:
            A wrapped AppConfig that automatically resolves session overrides when accessed.

        Raises:
            ConfigError: If no configuration is loaded.
        """
        if self._global_config is None:
            raise ConfigError(
                "No configuration loaded. Call load_global_config() first."
            )
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

        try:
            # Resolve path using PathManager and ensure parent directory exists
            resolved_path = self._path_manager.resolve_path(path, ensure_parent=True)

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

    @property
    def path_manager(self) -> PathManager:
        return self._path_manager
