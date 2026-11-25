"""Tool registry for managing tool definitions and implementations.

The registry is an injected application service, not a process-global singleton.
It stores both ToolDefinition metadata and ToolImplementation factories.
"""

from __future__ import annotations

from local_coding_assistant.tools_platform.contracts import ToolDefinition
from local_coding_assistant.tools_platform.definitions import ToolImplementation


class ToolRegistry:
    """Registry for tool definitions and implementations (injected, not singleton)."""

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._definitions: dict[str, ToolDefinition] = {}
        self._implementations: dict[str, ToolImplementation] = {}

    def register(
        self,
        definition: ToolDefinition,
        implementation: ToolImplementation,
    ) -> None:
        """Register a tool with its definition and implementation.

        Args:
            definition: Tool definition
            implementation: Tool implementation

        Raises:
            ValueError: If tool ID is already registered
        """
        tool_id = definition.id
        if tool_id in self._definitions:
            raise ValueError(f"Tool '{tool_id}' is already registered")

        self._definitions[tool_id] = definition
        self._implementations[tool_id] = implementation

    def get_tool(self, tool_id: str) -> ToolDefinition | None:
        """Get a tool definition by ID.

        Args:
            tool_id: Tool identifier

        Returns:
            Tool definition or None if not found
        """
        return self._definitions.get(tool_id)

    def get_implementation(self, tool_id: str) -> ToolImplementation | None:
        """Get a tool implementation by ID.

        Args:
            tool_id: Tool identifier

        Returns:
            Tool implementation or None if not found
        """
        return self._implementations.get(tool_id)

    def list_tools(self) -> list[ToolDefinition]:
        """List all registered tool definitions.

        Returns:
            List of tool definitions
        """
        return list(self._definitions.values())

    def get_tool_and_implementation(
        self,
        tool_id: str,
    ) -> tuple[ToolDefinition, ToolImplementation] | None:
        """Get both definition and implementation for a tool.

        Args:
            tool_id: Tool identifier

        Returns:
            Tuple of (definition, implementation) or None if not found
        """
        definition = self._definitions.get(tool_id)
        implementation = self._implementations.get(tool_id)

        if definition is None or implementation is None:
            return None

        return definition, implementation

    def has_tool(self, tool_id: str) -> bool:
        """Check if a tool is registered.

        Args:
            tool_id: Tool identifier

        Returns:
            True if tool is registered
        """
        return tool_id in self._definitions

    def clear(self) -> None:
        """Clear all registered tools (useful for testing)."""
        self._definitions.clear()
        self._implementations.clear()
