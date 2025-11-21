"""Protocols (interfaces) for core components.

This module defines abstract interfaces (using Protocol) that establish contracts
between different components of the system, helping to reduce coupling and
improve testability.
"""

from collections.abc import AsyncIterator, Iterable
from typing import Any, Protocol, runtime_checkable

from local_coding_assistant.tools.base import Tool
from local_coding_assistant.tools.types import (
    ToolCategory,
    ToolExecutionRequest,
    ToolExecutionResponse,
    ToolInfo,
)


@runtime_checkable
class IConfigManager(Protocol):
    """Protocol defining the configuration interface used by ToolManager.

    This protocol defines the minimal interface required by components that need
    to interact with the configuration system.
    """

    @property
    def global_config(self) -> Any | None:
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

    def load_global_config(self) -> Any:
        """Load and validate the global configuration.

        Returns:
            The loaded and validated configuration.

        Raises:
            ConfigError: If configuration is invalid or files don't exist
        """
        ...

    def get_tools(self) -> dict[str, Any]:
        """Get all configured tools.

        Returns:
            Dictionary mapping tool names to their configuration data.
        """
        ...

    def reload_tools(self) -> None:
        """Reload tools configuration from the source.

        This should refresh the internal state of the config manager
        with the latest configuration from the source.
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

    def resolve(
        self,
        provider: str | None = None,
        model_name: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> Any:
        """Resolve configuration with all layers applied.

        Args:
            provider: Optional provider override for this call
            model_name: Optional model_name override for this call
            overrides: Optional additional overrides for this call

        Returns:
            Resolved configuration with all layers merged

        Raises:
            ConfigError: If no global config is loaded or resolution fails
        """
        ...


@runtime_checkable
class IToolManager(Iterable[Any], Protocol):
    """Interface for tool management functionality.

    This protocol defines the interface for components that need to manage
    and interact with tools in the system. Implements Iterable to support
    iteration over available tools.
    """

    def register_tool(self, tool: Tool) -> None:
        """Register a tool with the manager."""
        ...

    def get_tool(self, name: str) -> Tool:
        """Get a tool by name."""
        ...

    def list_tools(self, category: str | ToolCategory | None = None) -> list[ToolInfo]:
        """List all registered tools, optionally filtered by category.

        Args:
            category: Optional category to filter tools by (can be string or ToolCategory).
                     If the category doesn't exist, returns an empty list.

        Returns:
            List of ToolInfo objects.
        """
        ...

    def execute(self, request: ToolExecutionRequest) -> ToolExecutionResponse:
        """Execute a tool using a ToolExecutionRequest.

        Args:
            request: ToolExecutionRequest containing tool name and payload.

        Returns:
            ToolExecutionResponse with execution results.
        """
        ...

    def run_tool(self, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Synchronously execute a tool.

        Args:
            tool_name: Name of the tool to execute
            payload: Input parameters for the tool

        Returns:
            The tool's execution result as a dictionary

        Raises:
            ToolRegistryError: If the tool is not found, is async, or execution fails
        """
        ...

    async def arun_tool(
        self, tool_name: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Asynchronously execute a tool.

        Args:
            tool_name: Name of the tool to execute
            payload: Input parameters for the tool

        Returns:
            The tool's execution result as a dictionary

        Raises:
            ToolRegistryError: If the tool is not found, is not async, or execution fails
        """
        ...

    def stream_tool(
        self, tool_name: str, payload: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream results from a tool that supports streaming.

        Args:
            tool_name: Name of the tool to execute
            payload: Input parameters for the tool

        Yields:
            Chunks of tool output as dictionaries

        Raises:
            ToolRegistryError: If the tool is not found, doesn't support streaming, or execution fails
        """
        ...
