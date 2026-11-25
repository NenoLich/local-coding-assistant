"""Workspace admission middleware for path scope and transaction validation.

This module implements the workspace admission middleware that checks
path scope, transaction state, and conflicts before allowing a call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from local_coding_assistant.tools_platform.contracts import (
    ValidatedToolCall,
)
from local_coding_assistant.tools_platform.exceptions import (
    ToolConflictError,
    ToolPathViolationError,
)
from local_coding_assistant.tools_platform.middleware.base import (
    MiddlewareContext,
    PrepareMiddleware,
)

if TYPE_CHECKING:
    from local_coding_assistant.tools_platform.workspace import WorkspaceManager


class WorkspaceAdmissionMiddleware(PrepareMiddleware):
    """Middleware for workspace admission checks.

    This middleware:
    - Checks path scope
    - Validates transaction state
    - Detects conflicts before allowing a call
    """

    def __init__(self, workspace_manager: WorkspaceManager) -> None:
        """Initialize workspace admission middleware.

        Args:
            workspace_manager: Workspace and transaction manager
        """
        self.workspace_manager = workspace_manager

    async def prepare(self, context: MiddlewareContext) -> None:
        """Check path scope, transaction state, and conflicts.

        Args:
            context: Middleware context

        Raises:
            ToolPathViolationError: If path validation fails
            ToolConflictError: If conflict detected
        """
        validated_call = context.validated_call
        if validated_call is None:
            raise ValueError("Validated call not set in context")

        await self._admit_to_workspace(validated_call)

    async def _admit_to_workspace(self, validated_call: ValidatedToolCall) -> None:
        """Check path scope, transaction state, and conflicts.

        Args:
            validated_call: Validated tool call

        Raises:
            ToolPathViolationError: If path validation fails
            ToolConflictError: If conflict detected
        """
        request = validated_call.request

        # Check workspace ID matches
        if request.workspace_id != self.workspace_manager.workspace_id:
            raise ToolPathViolationError(
                "Tool call workspace does not match the active workspace",
                details={"workspace_id": request.workspace_id},
            )

        # Check transaction state if transaction ID is provided
        if request.transaction_id is not None:
            active_transaction_id = self.workspace_manager.get_active_transaction_id()
            if active_transaction_id != request.transaction_id:
                raise ToolConflictError(
                    "Tool call transaction is not active",
                    details={
                        "transaction_id": request.transaction_id,
                        "active_transaction_id": active_transaction_id,
                    },
                )

        # Validate path arguments
        input_model = validated_call.input_model
        if hasattr(input_model, "model_dump"):
            arguments = input_model.model_dump()
        else:
            arguments = dict(input_model)

        self._validate_path_arguments(arguments)

    def _validate_path_arguments(self, arguments: dict[str, Any]) -> None:
        """Validate path arguments against workspace policy.

        Args:
            arguments: Tool arguments

        Raises:
            ToolPathViolationError: If any path argument violates policy
        """
        # Validate single path arguments
        for key in ("path", "cwd"):
            value = arguments.get(key)
            if isinstance(value, str):
                try:
                    self.workspace_manager.resolve_path(value)
                except ToolPathViolationError as exc:
                    raise ToolPathViolationError(
                        f"Path argument '{key}' violates workspace policy",
                        details={"path": value, "argument": key},
                    ) from exc

        # Validate multiple paths (e.g., in search tools)
        for value in arguments.get("paths", []):
            if isinstance(value, str):
                try:
                    self.workspace_manager.resolve_path(value)
                except ToolPathViolationError as exc:
                    raise ToolPathViolationError(
                        "A path argument violates workspace policy",
                        details={"path": value, "argument": "paths"},
                    ) from exc
