"""Tool registration system for automatic discovery and management of tools."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

from .base import Tool
from .types import ToolCategory, ToolInfo, ToolPermission, ToolSource, ToolTag

T = TypeVar("T", bound=Tool)


@dataclass
class ToolRegistration:
    """Model for registering tools with metadata.

    Attributes:
        name: Unique name of the tool
        tool_class: The tool class to register
        description: Human-readable description of the tool
        category: Category the tool belongs to
        source: Source of the tool (builtin, external, etc.)
        permissions: List of permissions required by the tool
        tags: List of tags for the tool
        is_async: Whether the tool's run method is async
        supports_streaming: Whether the tool supports streaming output
    """

    name: str
    tool_class: type[Tool]
    description: str = ""
    category: ToolCategory | None = None
    source: ToolSource = ToolSource.BUILTIN
    permissions: list[ToolPermission] = field(default_factory=list)
    tags: list[ToolTag] = field(default_factory=list)
    is_async: bool = False
    supports_streaming: bool = False

    def to_tool_info(self) -> ToolInfo:
        """Convert this registration to a ToolInfo object.

        Returns:
            ToolInfo: A ToolInfo instance with the registration data
        """
        # Convert enums to strings for ToolInfo compatibility
        permissions = [p.value if hasattr(p, "value") else p for p in self.permissions]
        tags = [t.value if hasattr(t, "value") else t for t in self.tags]

        return ToolInfo(
            name=self.name,
            tool_class=self.tool_class,
            description=self.description,
            category=self.category,
            source=self.source,
            permissions=permissions,  # Now a list of strings
            tags=tags,  # Now a list of strings
            is_async=self.is_async,
            supports_streaming=self.supports_streaming,
        )


# Global registry
_TOOL_REGISTRY: dict[str, ToolRegistration] = {}


def register_tool(
    *,
    name: str | None = None,
    description: str | None = None,
    category: ToolCategory | str | None = None,
    source: ToolSource = ToolSource.BUILTIN,
    permissions: list[ToolPermission | str] | None = None,
    tags: list[ToolTag | str] | None = None,
    is_async: bool | None = None,
    supports_streaming: bool = False,
) -> Callable[[type[T]], type[T]]:
    """Decorator to register a tool class with metadata.

    Args:
        name: Tool name (defaults to class name in snake_case)
        description: Tool description
        category: Tool category (from ToolCategory or string)
        source: Source of the tool (builtin, external, mcp)
        permissions: List of required permissions
        tags: List of tags for the tool
        is_async: Whether the tool's run method is async (auto-detected if None)
        supports_streaming: Whether the tool supports streaming output

    Returns:
        Decorator function
    """

    def decorator(tool_class: type[T]) -> type[T]:
        nonlocal name, description, category, permissions, tags, is_async

        # Default values
        if name is None:
            name = _camel_to_snake(tool_class.__name__)
        if description is None:
            description = tool_class.__doc__ or ""
        if category is not None and isinstance(category, str):
            category = ToolCategory(category)
        if permissions is None:
            permissions = []
        if tags is None:
            tags = []

        # Convert string permissions/tags to enums
        permissions = [
            ToolPermission(p) if isinstance(p, str) else p for p in permissions
        ]
        tags = [ToolTag(t) if isinstance(t, str) else t for t in tags]

        # Auto-detect async if not specified
        if is_async is None:
            run_method = getattr(tool_class, "run", None)
            is_async = inspect.iscoroutinefunction(run_method)

        # Create registration
        registration = ToolRegistration(
            name=name,
            tool_class=tool_class,
            description=description,
            category=category,
            source=source,
            permissions=permissions,
            tags=tags,
            is_async=is_async,
            supports_streaming=supports_streaming,
        )

        # Register the tool
        _TOOL_REGISTRY[name] = registration
        return tool_class

    return decorator


def get_tool_registry() -> dict[str, ToolRegistration]:
    """Get a copy of the tool registry."""
    return _TOOL_REGISTRY.copy()


def get_tool(name: str) -> Tool | None:
    """Get a tool instance by name."""
    registration = _TOOL_REGISTRY.get(name)
    if registration:
        return registration.tool_class()
    return None


def get_tools_by_category(category: ToolCategory | str) -> dict[str, ToolRegistration]:
    """Get all tools in a specific category."""
    if isinstance(category, str):
        category = ToolCategory(category)
    return {
        name: reg for name, reg in _TOOL_REGISTRY.items() if reg.category == category
    }


def get_tools_with_permission(
    permission: ToolPermission | str,
) -> dict[str, ToolRegistration]:
    """Get all tools that require a specific permission."""
    if isinstance(permission, str):
        permission = ToolPermission(permission)
    return {
        name: reg
        for name, reg in _TOOL_REGISTRY.items()
        if permission in reg.permissions
    }


def _camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case."""
    import re

    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


# Import all tools to register them
# This must be at the end to avoid circular imports
try:
    from .builtin_tools import *  # noqa
    from .mcp_tools import *  # noqa
except ImportError:
    # Some modules might not be available
    pass
