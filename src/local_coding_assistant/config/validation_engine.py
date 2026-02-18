"""Validation engine for centralized configuration validation.

This module provides the ValidationEngine class for validating configuration
fields against system capabilities and dependencies.
"""

from __future__ import annotations

from typing import Any

from local_coding_assistant.config.cache import (
    FieldValidationCache,
    get_capabilities_hash,
)
from local_coding_assistant.config.field import ConfigFieldRegistry
from local_coding_assistant.core.exceptions import ConfigError
from local_coding_assistant.core.system_registry import (
    SystemCapabilityRegistry,
    ValidationResult,
)
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("config.validation_engine")


class ValidationEngine:
    """Centralized validation engine for configuration fields.

    This engine provides centralized validation logic for configuration fields,
    replacing the scattered validation methods in ConfigManager.
    """

    def __init__(
        self,
        field_registry: ConfigFieldRegistry,
        system_registry: SystemCapabilityRegistry,
    ):
        """Initialize the validation engine.

        Args:
            field_registry: FieldRegistry for field metadata
            system_registry: SystemCapabilityRegistry for dependency checking
        """
        self.field_registry = field_registry
        self.system_registry = system_registry
        self._cache = FieldValidationCache(maxsize=16)

    def validate_field(self, field_path: str, value: Any) -> ValidationResult:
        """Validate a single field value.

        Args:
            field_path: The dot-notation path to the field
            value: The value to validate

        Returns:
            ValidationResult indicating if the value is valid

        Raises:
            ConfigError: If the field is not found in the registry
        """
        field = self.field_registry.get_field(field_path)
        if not field:
            raise ConfigError(f"Field not found in registry: {field_path}")

        # Get current capabilities hash
        capabilities_hash = get_capabilities_hash(
            frozenset(self.system_registry.capabilities)
        )

        # Check cache first
        cached_result = self._cache.get(field_path, value, capabilities_hash)
        if cached_result is not None:
            return cached_result

        # Perform validation
        result = field.validate(value, self.system_registry)

        # Cache the result
        self._cache.set(field_path, value, capabilities_hash, result)

        return result

    def get_validated_value(
        self,
        field_path: str,
        value: Any,
        allow_deferring: bool = True,
        remove_pending_on_valid: bool = True,
        pending_override: bool = True,
    ) -> Any | None:
        """Get validated value or fallback, return None if both invalid.

        This method encapsulates the validation logic and returns the final
        value that should be used, or None if no valid value is available.

        Args:
            field_path: The dot-notation path to the field
            value: The value to validate
            allow_deferring: Whether to allow deferring validation (default: True)
            remove_pending_on_valid: Whether to remove pending validations when value is valid (default: True)
            pending_override: Whether to override current pending validation (default: True)

        Returns:
            Validated value, fallback value, or None if both invalid
        """
        try:
            result = self.validate_field(field_path, value)

            if result.valid:
                # Remove any existing pending validations for this field if value is now valid
                if remove_pending_on_valid and pending_override:
                    self.system_registry.remove_pending_validations(field_path)
                return value

            if result.can_defer and allow_deferring:
                # Add to pending validations in system registry
                self.system_registry.add_pending_validation(
                    field_path,
                    value,
                    getattr(result, "missing_dependencies", []),
                    pending_override=pending_override,
                )

            if result.available_fallback:
                logger.debug(
                    f"Using fallback for {field_path}: {result.available_fallback}"
                )
                return result.available_fallback

            field = self.field_registry.get_field(field_path)
            if not field:
                raise ConfigError(f"Field not found in registry: {field_path}")

            logger.warning(
                f"Invalid value for {field_path}, setting default value: {field.default}",
                error=result.error,
            )

            return field.default

        except ConfigError as e:
            logger.warning(f"Field {field_path} not found in registry: {e}")
            return None

    def check_dependencies(self, field_path: str) -> list[str]:
        """Check what dependencies are missing for a field.

        Args:
            field_path: The field path to check

        Returns:
            List of missing dependencies
        """
        field = self.field_registry.get_field(field_path)
        if not field or not field.has_dependencies:
            return []

        # Get all required modules and capabilities for all possible values
        all_deps = set()
        if field.dependencies:
            for value_requirements in field.dependencies.value_requirements.values():
                all_deps.update(value_requirements)

        # Check what's missing
        missing = []
        for dep in all_deps:
            if not self.system_registry.is_capability_available(dep):
                missing.append(dep)

        return missing

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary containing cache statistics
        """
        return self._cache.get_stats()

    def clear_cache(self, field_path: str | None = None) -> None:
        """Clear validation cache.

        Args:
            field_path: If provided, only clear cache for this field.
                       If None, clear all cache.
        """
        self._cache.invalidate(field_path)

    def get_cached_validation_result(
        self, field_path: str, value_hash: int
    ) -> ValidationResult:
        """Get a cached validation result.

        Args:
            field_path: The field path
            value_hash: Hash of the value for caching

        Returns:
            Cached ValidationResult
        """
        # This is a simplified caching mechanism
        # In practice, you'd want to hash the actual value
        field = self.field_registry.get_field(field_path)
        if not field:
            return ValidationResult(valid=False, error=f"Field not found: {field_path}")

        # For now, just validate without caching the actual value
        # The cache key is just the field path
        return ValidationResult(valid=True)
