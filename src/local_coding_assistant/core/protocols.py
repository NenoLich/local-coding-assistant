"""Protocols (interfaces) for core components.

This module defines abstract interfaces (using Protocol) that establish contracts
between different components of the system, helping to reduce coupling and
improve testability.
"""

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from local_coding_assistant.config.path_manager import PathManager
    from local_coding_assistant.config.schemas import AppConfig
    from local_coding_assistant.core.system_registry import SystemCapabilityRegistry


@runtime_checkable
class IConfigManager(Protocol):
    """Protocol defining the configuration interface used by ToolManager.

    This protocol defines the minimal interface required by components that need
    to interact with the configuration system.
    """

    @property
    def path_manager(self) -> "PathManager":
        """Get the path manager."""
        ...

    @property
    def global_config(self) -> "AppConfig":
        """Get the current global configuration.

        Returns:
            The current global configuration, or None if not loaded.
        """
        ...

    @property
    def session_overrides(self) -> dict[str, Any]:
        """Get the current session overrides.

        Returns:
            Dictionary of current session overrides.
        """
        ...

    @property
    def system_registry(self) -> "SystemCapabilityRegistry":
        """Get the system capability registry.

        Returns:
            The SystemCapabilityRegistry instance.
        """
        ...

    def load_global_config(self) -> Any:
        """Load and validate the global configuration.

        Returns:
            The loaded and validated configuration.

        Raises:
            ConfigError: If configuration is invalid or files don't exist
        """
        ...

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
        ...

    def register_capability(self, capabilities: list[str]) -> dict[str, Any]:
        """Register a capability directly.

        Args:
            capabilities: List of the capability to add

        Returns:
            Dictionary of setting names to their new values (only actual changes)
        """
        ...

    def unregister_capability(self, capabilities: list[str]) -> dict[str, Any]:
        """Unregister system capabilities and revalidate affected settings.

        Args:
            capabilities: List of capability names to unregister

        Returns:
            Dictionary of affected setting names to their current values
        """
        ...

    def set_session_overrides(self, overrides: dict[str, Any]) -> None:
        """Set session-level configuration overrides.

        Args:
            overrides: Dictionary of configuration overrides using dot notation
                      (e.g., {"llm.model_name": "gpt-4", "llm.temperature": 0.5})

        Raises:
            ConfigError: If the overrides are invalid
        """
        ...

    def persist_tool_policies_overrides(
        self, rule_id: str, outcome: str, description: str = ""
    ) -> bool:
        """Persist a project override to tool_policies.local.yaml.

        Args:
            rule_id: The base rule ID to override
            outcome: The new outcome (allow, deny, ask)
            description: Optional description

        Returns:
            True if successful, False otherwise
        """
        ...

    def get_provider_by_model(self, model_id: str) -> str:
        """Return the provider id for the given model id.

        Searches ``global_config.providers`` for a provider whose ``models``
        list contains an entry matching ``model_id``.

        Args:
            model_id: Model identifier (e.g. ``google/gemini-2.0-flash``).

        Returns:
            Provider id (e.g. ``google``) or empty string when not found.
        """
        ...
