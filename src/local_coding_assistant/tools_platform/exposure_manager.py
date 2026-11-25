"""Exposure manager for LLM-visible tool filtering.

Centralizes exposure policy logic used by both middleware and context manager.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from local_coding_assistant.core.protocols import IConfigManager
    from local_coding_assistant.tools_platform.contracts import (
        BackendType,
        ToolDefinition,
    )
    from local_coding_assistant.tools_platform.registry import ToolRegistry


class ExposureManager:
    """Manages tool exposure policy for a session.

    This class encapsulates all logic for determining which tools are
    exposed to the LLM based on backend compatibility and exposure policy.
    It is used by both the ExposureMiddleware (for runtime validation)
    and the ContextManager (for prompt schema generation).
    """

    def __init__(
        self,
        config_manager: IConfigManager,
    ) -> None:
        """Initialize exposure manager with config manager.

        Args:
            config_manager: Config manager for dynamic configuration resolution
        """
        self._config_manager = config_manager

    @property
    def backend_type(self) -> BackendType:
        """Get backend type from config."""
        from local_coding_assistant.tools_platform.contracts import BackendType

        if self._config_manager is None:
            return BackendType.HOST
        backend_str = self._config_manager.global_config.tools_platform.backend
        try:
            return BackendType(backend_str)
        except ValueError:
            return BackendType.HOST

    @property
    def exposure_policy(self) -> dict[str, Any]:
        """Get exposure policy from config."""
        if self._config_manager is None:
            return {}
        exposed_tools = self._config_manager.global_config.tools_platform.exposed_tools
        if isinstance(exposed_tools, list):
            return {"allowed_tools": list(exposed_tools)}
        elif exposed_tools == "auto":
            return {}
        return {}

    def is_exposed(self, tool_definition: ToolDefinition) -> bool:
        """Check if a tool is exposed for the current context.

        Args:
            tool_definition: Tool definition to check

        Returns:
            True if tool is exposed, False otherwise
        """
        # Check backend compatibility
        if self.backend_type not in tool_definition.supported_backends:
            return False

        # Check exposure policy
        return self._is_exposed_by_policy(tool_definition)

    def _is_exposed_by_policy(self, tool_definition: ToolDefinition) -> bool:
        """Check if tool is exposed by policy.

        Args:
            tool_definition: Tool definition to check

        Returns:
            True if tool is exposed, False otherwise
        """
        # If no policy, expose all backend-compatible tools
        if not self.exposure_policy:
            return True

        # Check if tool is explicitly disabled
        disabled_tools = self.exposure_policy.get("disabled_tools", [])
        if tool_definition.id in disabled_tools:
            return False

        # Check if tool is in allowed list (if specified)
        allowed_tools = self.exposure_policy.get("allowed_tools")
        if allowed_tools is not None:
            return tool_definition.id in allowed_tools

        # Check risk class policy
        risk_policy = self.exposure_policy.get("risk_policy", {})
        if risk_policy:
            allowed_risks = risk_policy.get("allowed_risks")
            if allowed_risks is not None:
                return tool_definition.risk_class in allowed_risks

        return True

    def get_exposed_tools(self, registry: ToolRegistry) -> list[ToolDefinition]:
        """Get all tools exposed for the current context.

        Args:
            registry: Tool registry

        Returns:
            List of exposed tool definitions
        """
        all_tools = registry.list_tools()
        return [tool for tool in all_tools if self.is_exposed(tool)]

    def get_exposed_tool_schemas(self, registry: ToolRegistry) -> list[dict[str, Any]]:
        """Get JSON schemas for exposed tools.

        This generates the tool schemas in the format expected by the LLM.

        Args:
            registry: Tool registry

        Returns:
            List of tool schemas
        """
        exposed_tools = self.get_exposed_tools(registry)
        return [
            {
                "id": tool.id,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in exposed_tools
        ]
