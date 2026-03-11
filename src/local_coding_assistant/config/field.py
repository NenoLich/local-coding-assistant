"""Configuration field management for the hybrid configuration system.

This module provides the ConfigField class and related utilities for managing
individual configuration fields with validation, dependencies, and metadata.
"""

from __future__ import annotations

from typing import Any, ClassVar, overload

from pydantic import BaseModel
from pydantic.fields import Field

from local_coding_assistant.config.dependencies import SettingDependency
from local_coding_assistant.core.system_registry import (
    SystemCapabilityRegistry,
    ValidationResult,
)
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("config.field")


class ConfigField:
    """Field wrapper that adds dependency validation to Pydantic fields.

    This class wraps Pydantic's Field function to add dependency validation
    and registry functionality while maintaining full compatibility.
    The wrapper is automatically converted to a Pydantic Field during model creation.
    """

    def __init__(
        self,
        default: Any = ...,
        description: str | None = None,
        dependencies: SettingDependency | None = None,
        **kwargs,
    ):
        """Initialize ConfigField.

        Args:
            default: Default value for the field
            description: Field description
            dependencies: SettingDependency for validation
            **kwargs: Additional Pydantic field arguments
        """
        # Store dependency information
        self._dependencies = dependencies
        self._field_name = None  # Will be set when used in model
        self._parent_model = None  # Will be set when used in model
        self._full_path = None  # Will be built from parent context
        self._section = None  # Will be set from ConfigModel section

        # Store Pydantic field parameters for later use
        self._default = default
        self._description = description
        self._kwargs = kwargs

    def to_field(self, **kwargs):
        """Create a Pydantic Field from this ConfigField.

        Args:
            **kwargs: Additional arguments to pass to Field()

        Returns:
            Pydantic Field instance
        """
        return Field(
            default=self._default,
            description=self._description,
            **self._kwargs,
            **kwargs,
        )

    @property
    def field_name(self) -> str | None:
        """Get the field name."""
        return self._field_name

    @property
    def parent_model(self) -> type[BaseModel] | None:
        """Get the parent model class."""
        return self._parent_model

    @property
    def full_path(self) -> str | None:
        """Get the full dot-notation path (e.g., 'runtime.tool_call_mode')."""
        return self._full_path

    @property
    def default(self) -> Any:
        """Get the default value."""
        return self._default

    @property
    def section(self) -> str | None:
        """Get the section this field belongs to."""
        return self._section

    @property
    def dependencies(self) -> SettingDependency | None:
        """Get field dependencies."""
        return self._dependencies

    @property
    def has_dependencies(self) -> bool:
        """Check if field has dependencies."""
        return self._dependencies is not None

    def get_fallback_values(self) -> list[str]:
        """Get fallback values."""
        if self._dependencies:
            return self._dependencies.fallback_order
        return []

    def apply_to_config(self, config: BaseModel, value: Any) -> None:
        """Apply a value to this field in the given configuration.

        Args:
            config: The configuration to modify
            value: The value to set
        """
        if not self._full_path:
            raise ValueError("ConfigField not properly initialized - missing full path")

        # Navigate to the parent object
        parts = self._full_path.split(".")
        current = config
        for part in parts[:-1]:
            current = getattr(current, part)

        # Set the final value
        setattr(current, parts[-1], value)

    def validate(
        self, value: Any, registry: SystemCapabilityRegistry
    ) -> ValidationResult:
        """Validate a value against field dependencies.

        Args:
            value: Value to validate
            registry: SystemCapabilityRegistry for validation

        Returns:
            ValidationResult
        """
        if not self.has_dependencies:
            return ValidationResult(valid=True)

        if not self._full_path:
            raise ValueError("ConfigField not properly initialized - missing full path")

        if not self._dependencies:
            raise ValueError(
                "ConfigField not properly initialized - missing dependencies"
            )

        value_requirements = self._dependencies.value_requirements
        if str(value) not in value_requirements:
            return ValidationResult(valid=True)

        return registry.validate_setting(self._full_path, value)

    def set_context(
        self,
        field_name: str,
        parent_model: type[BaseModel],
        full_path: str,
        section: str,
    ) -> None:
        """Set field context when used in a model.

        Args:
            field_name: The field name in the model
            parent_model: The parent model class
            full_path: The full dot-notation path
            section: The section this field belongs to
        """
        self._field_name = field_name
        self._parent_model = parent_model
        self._full_path = full_path
        self._section = section

    # Make ConfigField work like Pydantic Field
    def __class_getitem__(cls, item):
        """Support type annotations."""
        return cls

    def __repr__(self):
        return f"ConfigField(name={self._field_name}, path={self._full_path}, section={self._section}, dependencies={self._dependencies})"


class ConfigFieldRegistry:
    """Registry for tracking ConfigField instances across the AppConfig hierarchy.

    This provides easy resolution between setting names and their ConfigField instances.
    """

    _instance: ClassVar[ConfigFieldRegistry | None] = None
    _fields: ClassVar[dict[str, ConfigField]] = {}

    def __new__(cls) -> ConfigFieldRegistry:
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def register_field(self, path: str, field: ConfigField) -> None:
        """Register a ConfigField with its full path.

        Args:
            path: Full dot-notation path (e.g., "runtime.tool_call_mode")
            field: ConfigField instance
        """
        self._fields[path] = field

        # Register dependencies with system registry for pending validation resolution
        if field.has_dependencies:
            from local_coding_assistant.core.system_registry import (
                system_capability_registry,
            )

            system_capability_registry.register_config_field_dependencies(field)

    def get_field(self, path: str) -> ConfigField | None:
        """Get ConfigField by path.

        Args:
            path: Full dot-notation path

        Returns:
            ConfigField instance or None
        """
        return self._fields.get(path)

    def get_all_fields(self) -> dict[str, ConfigField]:
        """Get all registered fields.

        Returns:
            Dictionary of path -> ConfigField
        """
        return self._fields.copy()

    def get_fields_with_dependencies(self) -> list[ConfigField]:
        """Get all fields that have dependencies.

        Returns:
            List of ConfigField instances with dependencies
        """
        return [field for field in self._fields.values() if field.has_dependencies]

    def clear(self) -> None:
        """Clear all registered fields."""
        self._fields.clear()
        logger.debug("Cleared ConfigField registry")


# Global registry instance
config_field_registry = ConfigFieldRegistry()


@overload
def config_field[T](
    default: T,
    description: str | None = None,
    dependencies: SettingDependency | None = None,
    **kwargs,
) -> T: ...


@overload
def config_field(
    default: Any = ...,
    description: str | None = None,
    dependencies: SettingDependency | None = None,
    **kwargs,
) -> Any: ...


def config_field(
    default: Any = ...,
    description: str | None = None,
    dependencies: SettingDependency | None = None,
    **kwargs,
) -> ConfigField:
    """Create a ConfigField for use in AppConfig models.

    Args:
        default: Default value
        description: Field description
        dependencies: SettingDependency for validation
        **kwargs: Additional Pydantic field arguments

    Returns:
        ConfigField instance
    """
    return ConfigField(
        default=default, description=description, dependencies=dependencies, **kwargs
    )


class ConfigModelMeta(type(BaseModel)):
    """Metaclass that processes ConfigField definitions in models."""

    def __new__(cls, name, bases, namespace, section: str | None = None, **kwargs):
        # Store section for this model class
        if section is not None:
            namespace["_section"] = section

        # Process ConfigField definitions and register them
        fields_to_register = []

        for key, value in list(namespace.items()):
            if isinstance(value, ConfigField):
                # Store the field for registration after model is created
                fields_to_register.append((key, value))
                # Replace with actual Pydantic Field in namespace
                namespace[key] = value.to_field()

        # Create the model class
        model_class = super().__new__(cls, name, bases, namespace, **kwargs)

        # Register ConfigFields with full paths
        for field_name, field in fields_to_register:
            # Build full path using section
            full_path = cls.build_field_path(model_class, field_name, section)
            field.set_context(field_name, model_class, full_path, section or "")
            config_field_registry.register_field(full_path, field)

        return model_class

    @staticmethod
    def build_field_path(
        model_class: Any, field_name: str, section: str | None = None
    ) -> str:
        """Build the full path for a field.

        Args:
            model_class: The model class containing the field
            field_name: The field name
            section: The section this model belongs to

        Returns:
            Full dot-notation path
        """
        if section:
            return f"{section}.{field_name}"

        # Fallback to model name if no section
        model_name = model_class.__name__.lower().replace("config", "")
        return f"{model_name}.{field_name}"

    @staticmethod
    def build_field_prefix(model_class: type, section: str | None = None) -> str:
        """Build the prefix for fields in a model.

        Args:
            model_class: The model class
            section: The section this model belongs to

        Returns:
            Prefix string (with trailing dot)
        """
        if section:
            return f"{section}."

        # Fallback to model name if no section
        model_name = model_class.__name__.lower().replace("config", "")
        return f"{model_name}."


class ConfigModel(BaseModel, metaclass=ConfigModelMeta):
    """Base model that supports ConfigField definitions."""

    _section: ClassVar[str | None] = None

    def __init_subclass__(cls, section: str | None = None, **kwargs: Any):
        """Allow section to be specified in class definition.

        Args:
            section: Section name for this config model
            **kwargs: Additional arguments
        """
        super().__init_subclass__(**kwargs)
        if section is not None:
            cls._section = section

    @classmethod
    def get_section(cls) -> str | None:
        """Get the section for this model."""
        return cls._section

    def get_config_field(self, field_name: str) -> ConfigField | None:
        """Get ConfigField by field name.

        Args:
            field_name: The field name within this model

        Returns:
            ConfigField instance or None
        """
        # Build full path and get from registry
        full_path = ConfigModelMeta.build_field_path(
            self.__class__, field_name, self._section
        )
        return config_field_registry.get_field(full_path)

    def get_fields_with_dependencies(self) -> list[ConfigField]:
        """Get all fields in this model that have dependencies.

        Returns:
            List of ConfigField instances with dependencies
        """
        prefix = ConfigModelMeta.build_field_prefix(self.__class__, self._section)

        fields = []

        for path, field in config_field_registry.get_all_fields().items():
            if path.startswith(prefix) and field.has_dependencies:
                fields.append(field)

        return fields
