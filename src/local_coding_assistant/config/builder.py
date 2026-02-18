"""Configuration builder for applying overrides and building configurations.

This module provides the ConfigBuilder class for building configurations
with override layers and validation using the ConfigField system.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from local_coding_assistant.config.field import config_field_registry
from local_coding_assistant.config.schemas import AppConfig
from local_coding_assistant.config.validation_engine import ValidationEngine
from local_coding_assistant.core.exceptions import ConfigError
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("config.builder")


class ConfigBuilder:
    """Builds final configuration with layered overrides and validation.

    This builder handles the application of session-level and call-level overrides
    to a base configuration, validates fields using the ConfigField system,
    and caches the built configuration for efficiency.
    """

    def __init__(
        self,
        base_config: AppConfig,
        validation_engine: ValidationEngine,
        allow_deferring: bool = True,
    ):
        """Initialize the config builder.

        Args:
            base_config: The base configuration to build upon
            validation_engine: ValidationEngine for field validation
            allow_deferring: Whether to allow deferring validation (default: True)
        """
        self._base_config = base_config
        self._validation_engine = validation_engine
        self._allow_deferring = allow_deferring
        self._session_overrides: dict[str, Any] = {}  # Just store validated values
        self._call_overrides: dict[str, Any] = {}  # Just store validated values

        # Cached built config
        self._built_config: AppConfig | None = None

        # Validate base config fields with dependencies
        self._is_base_config_validated = False

    def _validate_base_config(self) -> None:
        """Validate all fields with dependencies in the base config.

        Raises:
            ConfigError: If base config validation fails critically
        """
        fields_with_deps = config_field_registry.get_fields_with_dependencies()

        critical_failures = []

        for field in fields_with_deps:
            try:
                if field.full_path is None:
                    raise RuntimeError(
                        "Some fields in fields_with_deps do not have a full_path set."
                    )

                # Get current value from base config
                current_value = self._get_field_value(
                    self._base_config, field.full_path
                )

                # Get validated value or fallback from ValidationEngine
                validated_value = self._validation_engine.get_validated_value(
                    field.full_path,
                    current_value,
                    allow_deferring=self._allow_deferring,
                    pending_override=True,
                )

                if validated_value is not None and validated_value != current_value:
                    # Add fallback to session overrides (lower priority than explicit session overrides)
                    self._session_overrides[field.full_path] = validated_value
                    logger.info(
                        f"Added base config fallback to session overrides for {field.full_path}: {validated_value}"
                    )

                elif validated_value is None:
                    critical_failures.append(field.full_path)

            except Exception as e:
                logger.error(
                    f"Error validating base config field {field.full_path}: {e}"
                )
                critical_failures.append(field.full_path)

        if critical_failures:
            raise ConfigError(
                f"Base config validation failed for fields: {critical_failures}"
            )

        self._is_base_config_validated = True

    def _get_field_value(self, config: AppConfig, field_path: str) -> Any:
        """Get value from config using dot notation.

        Args:
            config: AppConfig instance
            field_path: Dot-notation path (e.g., "runtime.tool_call_mode")

        Returns:
            Field value
        """
        parts = field_path.split(".")
        current = config
        for part in parts:
            current = getattr(current, part)
        return current

    def set_session_overrides(
        self,
        overrides: dict[str, Any],
        skip_validation: bool = False,
        pending_override: bool = True,
    ) -> None:
        """Set session-level overrides from dict input (runtime from other modules).

        Args:
            overrides: Dictionary of field path to value overrides
            skip_validation: Whether to skip validation (default: False)
            pending_override: Whether to override current pending validation (default: True)
        """
        if not overrides:
            return

        logger.debug("Setting session overrides", overrides=overrides)

        for field_path, value in overrides.items():
            if skip_validation:
                validated_value = value
            else:
                # Get validated value or fallback from ValidationEngine
                validated_value = self._validation_engine.get_validated_value(
                    field_path,
                    value,
                    allow_deferring=self._allow_deferring,
                    pending_override=pending_override,
                )

            if validated_value is not None:
                self._session_overrides[field_path] = validated_value
                logger.debug(
                    f"Applied session override for {field_path}: {validated_value}"
                )
            else:
                logger.warning(
                    f"Skipping invalid session override for {field_path}: {value}"
                )

        # Invalidate cache
        self._built_config = None

    def set_call_overrides(self, overrides: dict[str, Any]) -> AppConfig:
        """Set call-level overrides from dict input.

        Args:
            overrides: Dictionary of field path to value overrides
        """
        logger.debug(f"Setting call overrides: {list(overrides.keys())}")

        for field_path, value in overrides.items():
            # Get validated value or fallback from ValidationEngine
            validated_value = self._validation_engine.get_validated_value(
                field_path, value, allow_deferring=self._allow_deferring
            )

            if validated_value is not None:
                self._call_overrides[field_path] = validated_value
                logger.debug(
                    f"Applied call override for {field_path}: {validated_value}"
                )
            else:
                logger.warning(
                    f"Skipping invalid call override for {field_path}: {value}"
                )

        # Invalidate cache
        self._built_config = None

        config = self.build()
        self.clear_call_overrides()
        self._built_config = None

        return config

    def clear_session_overrides(self) -> None:
        """Clear session overrides."""
        self._session_overrides.clear()
        self._built_config = None
        logger.debug("Cleared session overrides")

    def clear_call_overrides(self) -> None:
        """Clear call overrides."""
        self._call_overrides.clear()
        self._built_config = None
        logger.debug("Cleared call overrides")

    def build(self) -> AppConfig:
        """Build config applying validated overrides.

        Returns:
            Built AppConfig instance with all overrides applied
        """
        if self._built_config is None:
            self._built_config = self._build_config()

        return self._built_config

    def _build_config(self) -> AppConfig:
        """Internal build method that actually creates the configuration.

        Returns:
            Built AppConfig instance
        """
        # Start with base config
        if not self._is_base_config_validated:
            self._validate_base_config()

        config = deepcopy(self._base_config)

        # Create a dict representation of current config
        current_dict = config.model_dump()

        # Apply session overrides (includes base config fallbacks)
        for field_path, value in self._session_overrides.items():
            self._set_nested_value(current_dict, field_path, value)

        # Apply call overrides (higher priority than session)
        for field_path, value in self._call_overrides.items():
            self._set_nested_value(current_dict, field_path, value)

        # Validate the updated dict and update config
        if self._session_overrides or self._call_overrides:
            validated_config = config.__class__.model_validate(current_dict)
            config.__dict__.update(validated_config.__dict__)

        return config

    def _set_nested_value(self, data: dict, field_path: str, value: Any) -> None:
        """Set a nested value in a dict using dot notation."""
        parts = field_path.split(".")
        current = data
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    def get_session_overrides(self) -> dict[str, Any]:
        """Get copy of current session_overrides.

        Returns:
            Current session_overrides copy.
        """
        return deepcopy(self._session_overrides)

    def update_base_config(
        self, update_data: Any, field_path: str | None = None
    ) -> None:
        """Update the base configuration with new data.

        This method updates the base configuration with new data and invalidates
        the cached built configuration to ensure subsequent builds use the updated data.

        Args:
            update_data: New data to update in base config. Can be:
                - ConfigModel instance: Updates matching fields
                - dict: Updates using dict update semantics
                - Any other type: Assigned directly if field_path is specified
            field_path: Optional dot-notation path for direct assignment.
                       Required if update_data is not a ConfigModel or dict.

        Examples:
            # Update tools (ConfigModel)
            builder.update_base_config(tool_config_list)

            # Update nested field (dict)
            builder.update_base_config({"temperature": 0.8}, "llm")

            # Update specific field (direct assignment)
            builder.update_base_config("gpt-4", "llm.model_name")
        """
        if isinstance(update_data, type(self._base_config)):
            # Same type as base config - update matching fields
            update_dict = update_data.model_dump(exclude_unset=True)
            for field_name, value in update_dict.items():
                setattr(self._base_config, field_name, value)

        elif isinstance(update_data, dict):
            if field_path:
                # Update nested field using dot notation
                current = self._base_config
                parts = field_path.split(".")

                # Navigate to parent
                for part in parts[:-1]:
                    current = getattr(current, part)

                # Update the final field
                setattr(current, parts[-1], update_data)
            else:
                # Direct dict update of base config
                for field_name, value in update_data.items():
                    if hasattr(self._base_config, field_name):
                        setattr(self._base_config, field_name, value)
                    else:
                        logger.warning(f"Field '{field_name}' not found in base config")

        elif field_path:
            # Direct assignment to specific field path
            current = self._base_config
            parts = field_path.split(".")

            # Navigate to parent
            for part in parts[:-1]:
                current = getattr(current, part)

            # Update the final field
            setattr(current, parts[-1], update_data)
        else:
            raise ValueError(
                "field_path is required when update_data is not a ConfigModel or dict"
            )

        # Invalidate cached config and validation since base config changed
        self._built_config = None
        self._is_base_config_validated = False

        logger.debug(
            f"Updated base configuration. Data type: {type(update_data).__name__}, "
            f"Field path: {field_path or 'direct update'}"
        )

    def update_tools(self, tools: dict[str, Any]) -> None:
        """Update the tools configuration with loaded tool configurations.

        Args:
            tools: Dictionary of tool configurations keyed by tool ID

        Raises:
            ConfigError: If the base config doesn't have a tools field
        """
        if not hasattr(self._base_config, "tools"):
            raise ConfigError("Base config does not have a tools field")

        # Convert dict to ToolConfigList
        from local_coding_assistant.config.schemas import ToolConfigList

        tool_config_list = ToolConfigList(tools=list(tools.values()))

        # Update the tools field
        self._base_config.tools = tool_config_list

        # Invalidate cached config and validation since base config changed
        self._built_config = None
        self._is_base_config_validated = False
