"""Exposure middleware for LLM-visible tool filtering.

This module implements the exposure middleware that filters tools
based on backend compatibility, exposure policy, and session capabilities.
Hidden tools are absent from the prompt, not just rejected at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from local_coding_assistant.tools_platform.contracts import ToolDefinition
from local_coding_assistant.tools_platform.exceptions import ToolValidationError
from local_coding_assistant.tools_platform.exposure_manager import ExposureManager
from local_coding_assistant.tools_platform.middleware.base import (
    MiddlewareContext,
    PrepareMiddleware,
)

if TYPE_CHECKING:
    from local_coding_assistant.tools_platform.registry import ToolRegistry


class ExposureMiddleware(PrepareMiddleware):
    """Middleware for filtering tools visible to the LLM.

    This middleware:
    - Filters tools by backend compatibility (host vs Docker)
    - Applies exposure policy (mode, session, capabilities)
    - Removes hidden tools from prompt (not just reject at runtime)
    """

    def __init__(
        self,
        exposure_manager: ExposureManager,
    ) -> None:
        """Initialize exposure middleware.

        Args:
            exposure_manager: Exposure manager for the session
        """
        self._manager = exposure_manager

    @property
    def manager(self) -> ExposureManager:
        """Access the exposure manager for external use (e.g., by gateway)."""
        return self._manager

    async def prepare(self, context: MiddlewareContext) -> None:
        """Check if tool is exposed for current context.

        This validates:
        - Tool is registered
        - Tool supports the session's backend
        - Tool is enabled by exposure policy

        Args:
            context: Middleware context

        Raises:
            ToolValidationError: If tool is not exposed
        """
        tool_definition = await self._get_exposed_tool(
            context.registry,
            context.request.tool_id,
        )
        # Store tool definition in context for later middleware
        context.metadata["tool_definition"] = tool_definition

    async def _get_exposed_tool(
        self,
        registry: ToolRegistry,
        tool_id: str,
    ) -> ToolDefinition:
        """Get tool definition if it is exposed for current context.

        Args:
            registry: Tool registry
            tool_id: Tool identifier

        Returns:
            Tool definition if exposed

        Raises:
            ToolValidationError: If tool is not registered or not exposed
        """
        tool_definition = registry.get_tool(tool_id)
        if tool_definition is None:
            raise ToolValidationError(
                f"Tool '{tool_id}' is not registered",
            )

        if not self._manager.is_exposed(tool_definition):
            raise ToolValidationError(
                f"Tool '{tool_id}' is not exposed",
                details={
                    "tool_id": tool_id,
                    "backend": self._manager.backend_type.value,
                    "exposure_policy": self._manager.exposure_policy,
                },
            )

        return tool_definition
