"""System capability registry for tracking and validating system dependencies.

This module provides centralized tracking of system capabilities and
dependency validation for configuration settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import dropwhile
from typing import Any

from local_coding_assistant.utils.logging import get_logger

log = get_logger("core.system_registry")


@dataclass
class ValidationResult:
    """Result of setting validation."""

    valid: bool
    error: str | None = None
    can_defer: bool = False
    missing_dependencies: list[str] = field(default_factory=list)
    available_fallback: str | None = None
    original_error: str | None = None


@dataclass
class PendingValidation:
    """A setting validation that's pending module availability."""

    setting_name: str
    setting_value: str
    dependencies: list[str]
    timestamp: float = field(default_factory=lambda: __import__("time").time())

    def __hash__(self) -> int:
        """Make PendingValidation hashable for use in sets."""
        return hash((self.setting_name, self.setting_value, tuple(self.dependencies)))


class SystemCapabilityRegistry:
    """Central registry for tracking system capabilities and dependencies.

    This registry tracks:
    - Available system capabilities (python execution, sandbox, etc.)
    - Module initialization status
    - Pending setting validations waiting for modules
    - Dependency metadata for configuration settings
    """

    def __init__(self) -> None:
        """Initialize the system capability registry."""
        self.capabilities: set[str] = set()
        self.module_status: dict[str, bool] = {}
        self.pending_validations: list[PendingValidation] = []
        self.capability_to_settings: dict[str, set[tuple[str, Any]]] = {}

        # Initialize basic capabilities that don't require modules
        self._initialize_basic_capabilities()

    def _initialize_basic_capabilities(self) -> None:
        """Initialize capabilities that can be determined without modules."""
        import platform

        # Basic system info
        self.capabilities.add(platform.system())

        log.debug("Basic capabilities initialized", capabilities=self.capabilities)

    def register_capability(self, capabilities: list[str]) -> list[tuple[str, Any]]:
        """Register a capability directly.

        Args:
            capabilities: List of the capabilities

        Returns:
            List of (setting_name, setting_value) tuples for resolved validations
        """
        current_capabilities = self.capabilities.copy()
        self.capabilities.update(capabilities)

        if current_capabilities == self.capabilities:
            return []

        log.debug("Registered capabilities", capabilities=capabilities)

        resolved = self._resolve_pending_validations(capabilities=capabilities)

        return resolved

    def unregister_capability(self, capabilities: list[str]) -> list[str]:
        """Remove capabilities and return affected setting names.

        This method removes the specified capabilities from the system and
        returns a list of setting names that are affected by this change.
        The ConfigManager will then handle revalidation of these settings.

        Args:
            capabilities: List of the capabilities to remove

        Returns:
            List of affected setting names (not values)
        """
        current_capabilities = (
            self.capabilities.copy()
        )  # Make a copy to avoid reference issues
        self.capabilities.difference_update(set(capabilities))

        if current_capabilities == self.capabilities:
            return []

        log.info("Unregistered capabilities", capabilities=capabilities)

        # Find all affected settings (names only)
        affected_settings = set()
        for cap in capabilities:
            settings = self.capability_to_settings.get(cap, set())
            for setting_name, _ in settings:
                affected_settings.add(setting_name)

        log.debug(
            "Settings affected by capability removal: %s", list(affected_settings)
        )

        return list(affected_settings)

    def register_module(
        self, module_name: str, capabilities: dict[str, Any] | None = None
    ) -> list[tuple[str, Any]]:
        """Register a system module and its capabilities.

        Args:
            module_name: Name of the module (e.g., "tool_manager", "sandbox_manager")
            capabilities: Dictionary of capabilities provided by this module

        Returns:
            List of (setting_name, resolved_value) tuples for resolved pending validations
        """
        self.module_status[module_name] = True
        capabilities_list = [module_name]

        if capabilities:
            for cap in capabilities:
                if capabilities[cap]:
                    capabilities_list.append(cap)

        self.capabilities.update(capabilities_list)

        log.debug(
            "Module '%s' registered with capabilities: %s",
            module_name,
            capabilities_list,
        )

        # Resolve any pending validations that were waiting for this module
        resolved = self._resolve_pending_validations(module_name, capabilities_list)

        return resolved

    def register_config_field_dependencies(self, _field) -> None:
        """Register ConfigField dependencies for capability-to-settings mapping.

        This method maintains capability_to_settings mapping for pending validation
        resolution by extracting dependencies from ConfigField instances.

        Args:
            _field: ConfigField instance to register dependencies for
        """
        if not _field.has_dependencies:
            return

        # Extract dependencies from ConfigField
        value_requirements = _field.dependencies.value_requirements
        total_capabilities = 0

        for value, requirements in value_requirements.items():
            # requirements is now a unified list of modules and capabilities
            total_capabilities += len(requirements)
            for cap in requirements:
                self.capability_to_settings.setdefault(cap, set()).add(
                    (_field.full_path, value)
                )

        log.debug(
            "ConfigField dependencies registered for '%s': %d capabilities",
            _field.full_path,
            total_capabilities,
        )

    def validate_setting(
        self, setting_name: str, setting_value: Any
    ) -> ValidationResult:
        """Validate a setting against current system capabilities using ConfigField registry.

        Args:
            setting_name: Name of the setting to validate
            setting_value: Value to validate

        Returns:
            ValidationResult with validation status
        """
        try:
            # Lazy import to avoid circular dependency
            from local_coding_assistant.config.field import config_field_registry

            _field = config_field_registry.get_field(setting_name)

            if _field and _field.has_dependencies:
                # Use ConfigField dependencies for validation
                return self._validate_field_dependencies(_field, setting_value)
            else:
                # No dependencies defined, assume valid
                return ValidationResult(valid=True)

        except ImportError:
            # ConfigField registry not available, assume valid
            return ValidationResult(valid=True)

    def _validate_field_dependencies(
        self, _field, setting_value: Any
    ) -> ValidationResult:
        """Validate field dependencies using ConfigField metadata.
        Args:
            _field: ConfigField instance
            setting_value: Value to validate

        Returns:
            ValidationResult for the field
        """
        # Get value requirements from ConfigField dependencies
        setting_value = str(setting_value)
        value_requirements = _field.dependencies.value_requirements
        if str(setting_value) not in value_requirements:
            # Value doesn't have specific requirements, it's valid
            return ValidationResult(valid=True)

        required_items = value_requirements[str(setting_value)]

        # Check availability (modules in module_status, capabilities in capabilities)
        missing_items = []
        for item in required_items:
            if item not in self.capabilities:
                missing_items.append(item)

        if missing_items:
            # Try fallback values if available
            fallback_order = _field.dependencies.fallback_order
            # Drop everything until setting_value, then drop setting_value itself
            remaining = dropwhile(
                lambda x: str(x).lower() != setting_value.lower(), fallback_order
            )
            next(remaining, None)  # Skip setting_value
            fallback_value = next(remaining, None)
            log.debug(
                "Trying fallback value '%s' for setting '%s'",
                fallback_value,
                _field.full_path,
            )
            if fallback_value:
                # Check if fallback value would be valid
                fallback_result = self._validate_field_dependencies(
                    _field, fallback_value
                )
                if fallback_result.valid or fallback_result.available_fallback:
                    return ValidationResult(
                        valid=False,
                        available_fallback=fallback_result.available_fallback
                        or fallback_value,
                        can_defer=True,
                        missing_dependencies=missing_items,
                    )

            # No valid fallback found
            log.debug("No valid fallback found for setting '%s'", _field.full_path)
            return ValidationResult(
                valid=False,
                error=f"Missing dependencies: {missing_items}",
                can_defer=True,
                missing_dependencies=missing_items,
            )

        return ValidationResult(valid=True)

    def add_pending_validation(
        self,
        setting_name: str,
        setting_value: Any,
        dependencies: list[str],
        pending_override: bool = True,
    ) -> None:
        """Add or replace pending validation for when modules become available.

        This method ensures there's only one pending validation per setting
        by replacing any existing pending validation for the same setting.

        Args:
            setting_name: Name of the setting
            setting_value: Value to validate
            dependencies: List of dependencies that need to be available
            pending_override: Whether to override current pending validation (default: True)
        """

        if not pending_override:
            for pending in self.pending_validations:
                if setting_name == pending.setting_name:
                    log.debug(
                        f"Current pending validation {setting_name} with value: {pending.setting_value} preserved"
                    )
                    return

        # Remove any existing pending validation for this setting
        self.remove_pending_validations(setting_name)

        # Add new pending validation
        pending = PendingValidation(
            setting_name=setting_name,
            setting_value=str(setting_value),
            dependencies=dependencies,
        )
        self.pending_validations.append(pending)

        log.debug(
            "Added pending validation for setting '%s' with value '%s' waiting for: %s",
            setting_name,
            setting_value,
            dependencies,
        )

    def remove_pending_validations(self, setting_name: str) -> None:
        """Remove all pending validations for a specific setting.

        This is called when a setting becomes valid (e.g., user explicitly
        sets a valid value), so we don't override their choice when
        dependencies become available.

        Args:
            setting_name: Name of the setting to remove pending validations for
        """
        original_count = len(self.pending_validations)
        self.pending_validations = [
            pending
            for pending in self.pending_validations
            if pending.setting_name != setting_name
        ]
        removed_count = original_count - len(self.pending_validations)

        if removed_count > 0:
            log.debug(
                "Removed %d pending validation(s) for setting '%s'",
                removed_count,
                setting_name,
            )

    def _resolve_pending_validations(
        self, module_name: str | None = None, capabilities: list[str] | None = None
    ) -> list[tuple[str, Any]]:
        """Resolve pending validations that were waiting for a specific module.

        Args:
            module_name: Name of the module that just became available
            capabilities: List of capabilities that just became available

        Returns:
            List of (setting_name, resolved_value) tuples for resolved validations
        """
        resolved = []
        pending_validations = set()
        new_capabilities = set()

        if module_name:
            new_capabilities.add(module_name)

        if capabilities:
            new_capabilities.update(capabilities)

        for cap in new_capabilities:
            if cap in self.capability_to_settings:
                for pending in self.pending_validations:
                    if (
                        pending.setting_name,
                        pending.setting_value,
                    ) in self.capability_to_settings[cap]:
                        pending_validations.add(pending)

        for pending in pending_validations:
            # Try to validate this pending setting
            result = self.validate_setting(pending.setting_name, pending.setting_value)

            if result.valid:
                log.debug(
                    "Pending validation for '%s' resolved successfully",
                    pending.setting_name,
                )
                resolved.append((pending.setting_name, pending.setting_value))
                self.pending_validations.remove(pending)

            elif result.available_fallback:
                log.debug(
                    "Pending validation for '%s' failed due to missing dependencies: %s, fallback used: '%s'",
                    pending.setting_name,
                    result.missing_dependencies,
                    result.available_fallback,
                )
                resolved.append((pending.setting_name, result.available_fallback))
                # DON'T remove pending validation - keep it for future resolution when all dependencies are available

            else:
                log.debug(
                    "Pending validation for '%s' failed: %s",
                    pending.setting_name,
                    result.error,
                )

        return resolved

    def is_module_available(self, module_name: str) -> bool:
        """Check if a module is available and initialized.

        Args:
            module_name: Name of the module to check

        Returns:
            True if module is available, False otherwise
        """
        return self.module_status.get(module_name, False)

    def is_capability_available(self, capability: str) -> bool:
        """Check if a capability is available (alias for has_capability).

        Args:
            capability: Name of the capability to check

        Returns:
            True if capability is available, False otherwise
        """
        return capability in self.capabilities

    def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status for debugging.

        Returns:
            Dictionary containing all system status information
        """
        return {
            "capabilities": self.capabilities.copy(),
            "modules": self.module_status.copy(),
            "pending_validations": [
                {
                    "setting": pv.setting_name,
                    "value": pv.setting_value,
                    "dependencies": pv.dependencies,
                    "timestamp": pv.timestamp,
                }
                for pv in self.pending_validations
            ],
            "capability_to_settings": {
                cap: list(settings)
                for cap, settings in self.capability_to_settings.items()
            },
        }


# Global system registry instance
system_capability_registry = SystemCapabilityRegistry()
