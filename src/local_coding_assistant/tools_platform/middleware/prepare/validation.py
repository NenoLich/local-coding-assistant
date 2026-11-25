"""Validation middleware for argument and path normalization.

This module implements the validation middleware that validates
JSON schema, normalizes paths, applies defaults, and detects
dangerous arguments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from local_coding_assistant.tools_platform.contracts import (
    ToolCallRequest,
    ToolDefinition,
    ValidatedToolCall,
)
from local_coding_assistant.tools_platform.exceptions import (
    ToolPathViolationError,
    ToolValidationError,
)
from local_coding_assistant.tools_platform.middleware.base import (
    MiddlewareContext,
    PrepareMiddleware,
)

if TYPE_CHECKING:
    from local_coding_assistant.tools_platform.workspace import WorkspaceManager

# Per-tool caps derived from the global limits (max_output_bytes,
# max_read_lines, max_search_matches). Values are clamped so LLM-supplied
# arguments can never exceed the configured session bounds.
TOOL_INPUT_LIMITS: dict[str, dict[str, str]] = {
    "read_file": {"limit": "max_output_bytes"},
    "search_text": {
        "max_results": "max_search_matches",
    },
    "search_symbols": {"limit": "max_search_matches"},
    "match_paths": {"max_results": "max_search_matches"},
}


class ValidationMiddleware(PrepareMiddleware):
    """Middleware for argument and path normalization.

    This middleware:
    - Validates JSON shape using Pydantic models
    - Normalizes paths and resolves against workspace root
    - Applies default values
    - Clamps tool-specific inputs to configured limits
    - Detects dangerous arguments (e.g., absolute paths, shell escapes)
    """

    def __init__(
        self,
        workspace_manager: WorkspaceManager,
        limits: dict[str, int] | None = None,
    ) -> None:
        """Initialize validation middleware.

        Args:
            workspace_manager: Workspace manager for path normalization
            limits: Optional configured limits keyed by name
                (max_output_bytes, max_read_lines, max_search_matches)
        """
        self.workspace_manager = workspace_manager
        self.limits = limits or {}

    async def prepare(self, context: MiddlewareContext) -> None:
        """Validate and normalize tool call arguments.

        Args:
            context: Middleware context

        Raises:
            ToolValidationError: If validation fails
            ToolPathViolationError: If path violates workspace policy
        """
        # Get tool definition from exposure middleware
        tool_definition = context.metadata.get("tool_definition")
        if tool_definition is None:
            tool_definition = context.registry.get_tool(context.request.tool_id)
            if tool_definition is None:
                raise ToolValidationError(f"Tool not found: {context.request.tool_id}")

        validated_call = await self._validate_call(
            context.request,
            tool_definition,
            context.registry,
        )
        context.validated_call = validated_call

    async def _validate_call(
        self,
        request: ToolCallRequest,
        tool_definition: ToolDefinition,
        registry: Any,
    ) -> ValidatedToolCall:
        """Validate and normalize tool call arguments.

        Args:
            request: Tool call request
            tool_definition: Tool definition
            registry: Tool registry

        Returns:
            Validated tool call

        Raises:
            ToolValidationError: If validation fails
            ToolPathViolationError: If path violates workspace policy
        """
        # Get the tool implementation
        tool_impl = registry.get_implementation(request.tool_id)
        if tool_impl is None:
            raise ToolValidationError(
                f"Tool implementation not found: {request.tool_id}"
            )

        # Validate arguments using the tool's input model
        input_model = tool_impl.input_model.model_validate(request.raw_arguments)

        # Reject argument fields not declared by the tool's input model
        declared_fields = set(tool_impl.input_model.model_fields)
        extra_fields = set(request.raw_arguments.keys()) - declared_fields
        if extra_fields:
            raise ToolValidationError(
                f"Unexpected argument fields: {sorted(extra_fields)}",
                details={
                    "tool_id": tool_definition.id,
                    "extra_fields": sorted(extra_fields),
                },
            )

        # Normalize paths according to workspace policy
        normalized_paths = self._normalize_paths(input_model, tool_definition.id)
        input_updates = {
            key: value
            for key, value in normalized_paths.items()
            if not key.startswith("_")
        }

        # Clamp tool inputs to the configured session limits
        clamped = self._clamp_limits(input_model, tool_definition.id)
        input_updates.update(clamped)

        if input_updates:
            input_model = input_model.model_copy(update=input_updates)

        # Detect dangerous arguments
        self._check_dangerous_arguments(input_model, tool_definition.id)

        return ValidatedToolCall(
            request=request,
            tool_definition=tool_definition,
            input_model=input_model,
            normalized_paths=normalized_paths,
            execution_metadata={},
        )

    def _clamp_limits(
        self,
        input_model: BaseModel,
        tool_id: str,
    ) -> dict[str, Any]:
        """Clamp tool-specific inputs to the configured session limits.

        Args:
            input_model: Validated input model
            tool_id: Tool identifier

        Returns:
            Dictionary of fields to update with clamped values
        """
        tool_limits = TOOL_INPUT_LIMITS.get(tool_id)
        if not tool_limits or not self.limits:
            return {}

        arguments = (
            input_model.model_dump()
            if hasattr(input_model, "model_dump")
            else dict(input_model)
        )

        updates: dict[str, Any] = {}
        for field_name, limit_key in tool_limits.items():
            if field_name not in arguments:
                continue
            limit = self.limits.get(limit_key)
            value = arguments[field_name]
            if (
                limit is not None
                and isinstance(value, int)
                and not isinstance(value, bool)
                and value > limit
            ):
                updates[field_name] = limit
        return updates

    def _normalize_paths(
        self,
        input_model: BaseModel,
        tool_id: str,
    ) -> dict[str, Any]:
        """Normalize and validate paths in tool arguments.

        This implements the Workspace Boundary Filter by delegating to
        WorkspaceManager.normalize_tool_paths, which normalizes paths
        and marks out-of-workspace paths for boundary checks.

        Args:
            input_model: Validated input model
            tool_id: Tool identifier

        Returns:
            Dictionary mapping original path arguments to normalized paths

        Raises:
            ToolPathViolationError: If path violates workspace policy
        """
        arguments = (
            input_model.model_dump()
            if hasattr(input_model, "model_dump")
            else dict(input_model)
        )

        # Delegate to WorkspaceManager for canonical path normalization
        normalized = self.workspace_manager.normalize_tool_paths(arguments)

        # Check for out-of-workspace paths marked by WorkspaceManager
        if "_outside_workspace" in normalized:
            raise ToolPathViolationError(
                "Path argument is outside the workspace",
                details={"tool_id": tool_id, "arguments": arguments},
            )

        # Additional validation for specific path arguments
        for key in ("path", "cwd"):
            value = arguments.get(key)
            if not isinstance(value, str):
                continue
            try:
                resolved = self.workspace_manager.resolve_path(value)
            except ToolPathViolationError as exc:
                raise ToolPathViolationError(
                    f"Path argument '{key}' is outside the workspace",
                    details={"path": value, "argument": key},
                ) from exc
            normalized[key] = self.workspace_manager.get_relative_path(resolved)

        # Validate multiple paths (e.g., in search tools)
        for value in arguments.get("paths", []):
            if not isinstance(value, str):
                continue
            try:
                self.workspace_manager.resolve_path(value)
            except ToolPathViolationError as exc:
                raise ToolPathViolationError(
                    "A path argument is outside the workspace",
                    details={"path": value, "argument": "paths"},
                ) from exc

        return normalized

    def _check_dangerous_arguments(
        self,
        input_model: BaseModel,
        tool_id: str,
    ) -> None:
        """Check for dangerous argument patterns.

        This detects:
        - Absolute paths in relative-only contexts
        - Shell escape sequences
        - Command injection patterns

        Args:
            input_model: Validated input model
            tool_id: Tool identifier

        Raises:
            ToolValidationError: If dangerous arguments detected
        """
        arguments = (
            input_model.model_dump()
            if hasattr(input_model, "model_dump")
            else dict(input_model)
        )

        self._check_shell_escape_patterns(arguments)

        # Check for suspicious path patterns
        for key in ("path", "cwd"):
            value = arguments.get(key)
            if isinstance(value, str):
                # Check for path traversal attempts
                if ".." in value:
                    # Allow ../ only if it resolves within workspace
                    try:
                        resolved = self.workspace_manager.resolve_path(value)
                        self.workspace_manager.get_relative_path(resolved)
                    except ValueError as e:
                        raise ValueError(
                            f"Path traversal detected in argument '{key}': {value}"
                        ) from e

    def _check_shell_escape_patterns(
        self,
        arguments: dict[str, Any],
    ) -> None:
        """Check for shell escape patterns in string arguments.

        Fields holding arbitrary file content (e.g. ``content``, ``old_text``,
        ``new_text``) may legitimately contain shell-like characters and are
        skipped here. Query/search fields are also skipped since they may
        legitimately contain shell-like characters.
        """
        dangerous_patterns = ["&&", "||", ";", "|", "$(", "`", "$("]

        skip_fields = {"content", "old_text", "new_text", "query", "search"}
        for key, value in arguments.items():
            if key in skip_fields:
                continue
            values: list[str] = []
            if isinstance(value, str):
                values = [value]
            elif isinstance(value, list):
                values = [item for item in value if isinstance(item, str)]
            for item in values:
                for pattern in dangerous_patterns:
                    if pattern in item:
                        raise ValueError(
                            f"Dangerous pattern '{pattern}' detected in argument '{key}'"
                        )
