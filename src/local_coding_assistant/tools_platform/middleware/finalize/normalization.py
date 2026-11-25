"""Normalization middleware for backend/tool result conversion.

This module implements the normalization middleware that converts
backend-specific results to canonical ToolExecutionResult, normalizes
stdout/stderr across host and Docker, applies output truncation metadata,
and converts exceptions to canonical ToolError.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from local_coding_assistant.tools_platform.contracts import (
    BackendType,
    ToolExecutionError,
    ToolExecutionResult,
)
from local_coding_assistant.tools_platform.error_mapping import map_exception
from local_coding_assistant.tools_platform.exceptions import (
    ToolBackendError,
    ToolPlatformError,
)
from local_coding_assistant.tools_platform.middleware.base import (
    FinalizeMiddleware,
    MiddlewareContext,
)

if TYPE_CHECKING:
    from local_coding_assistant.tools_platform.backends.base import ExecutionBackend


class NormalizationMiddleware(FinalizeMiddleware):
    """Middleware for backend/tool result conversion.

    This middleware:
    - Converts backend-specific results to canonical ToolExecutionResult
    - Normalizes stdout/stderr across host and Docker
    - Applies output truncation metadata
    - Converts exceptions to canonical ToolError
    """

    def __init__(self, backend: ExecutionBackend) -> None:
        """Initialize normalization middleware.

        Args:
            backend: Execution backend for type information
        """
        self.backend = backend

    async def finalize(
        self, context: MiddlewareContext, result: ToolExecutionResult[Any]
    ) -> ToolExecutionResult[Any]:
        """Normalize the execution result.

        Args:
            context: Middleware context
            result: Execution result from backend

        Returns:
            Normalized execution result
        """
        # If result is already canonical, apply any additional normalization
        if isinstance(result, ToolExecutionResult):
            return self._normalize_canonical_result(result, context)

        # Convert raw backend result to canonical form
        return await self._convert_to_canonical(result, context)

    async def on_error(
        self, context: MiddlewareContext, error: Exception
    ) -> ToolExecutionError:
        """Convert exception to canonical ToolExecutionError.

        Args:
            context: Middleware context
            error: Exception that occurred

        Returns:
            Canonical tool execution error
        """
        # Convert platform exceptions to tool errors
        if isinstance(error, ToolPlatformError):
            return error.to_tool_error()

        # Map other exceptions using the error mapping service
        backend_type = self._get_backend_type()
        request = context.request
        return map_exception(
            error,
            tool_id=request.tool_id,
            call_id=request.call_id,
            backend=backend_type.value,
            workspace_id=request.workspace_id,
        )

    async def _convert_to_canonical(
        self,
        result: Any,
        context: MiddlewareContext,
    ) -> ToolExecutionResult[Any]:
        """Convert raw backend result to canonical ToolExecutionResult.

        Args:
            result: Raw result from backend
            context: Middleware context

        Returns:
            Canonical tool execution result
        """
        validated_call = context.validated_call
        if validated_call is None:
            raise ValueError("Validated call not set in context")

        # Validate against tool's output model if defined
        implementation = context.registry.get_implementation(
            validated_call.request.tool_id
        )
        if implementation is not None and implementation.output_model is not None:
            result = implementation.output_model.model_validate(result)

        # Apply output truncation metadata
        result = self._apply_truncation_metadata(result, context)

        return ToolExecutionResult(
            status="completed",
            output=result,
            backend=self._get_backend_type(),
        )

    def _normalize_canonical_result(
        self,
        result: ToolExecutionResult[Any],
        context: MiddlewareContext,
    ) -> ToolExecutionResult[Any]:
        """Normalize an already-canonical result.

        This applies any additional normalization like truncation metadata
        or backend-specific adjustments.

        Args:
            result: Canonical execution result
            context: Middleware context

        Returns:
            Normalized execution result
        """
        # Apply truncation metadata if not already present
        if result.output is not None:
            result.output = self._apply_truncation_metadata(result.output, context)

        return result

    def _apply_truncation_metadata(
        self,
        output: Any,
        context: MiddlewareContext,
    ) -> Any:
        """Apply truncation metadata to output.

        Records which stream was truncated, the byte/line counts before
        truncation, and whether output was dropped or clipped, in the
        result metadata so the LLM is told when output was cut.

        Args:
            output: Output data
            context: Middleware context

        Returns:
            Output with truncation metadata
        """
        # If output is a dict, add truncation metadata
        if isinstance(output, dict):
            metadata = output.setdefault("_truncation_metadata", {})
            self._populate_truncation_metadata(output, metadata, context)
        elif isinstance(output, BaseModel):
            # If output is a Pydantic model with a truncation field,
            # normalize a None value to False (best-effort).
            truncated = getattr(output, "truncated", None)
            if truncated is None:
                try:
                    setattr(output, "truncated", False)  # noqa: B010 - best-effort field set
                except Exception:  # noqa: S110 - best-effort field set
                    pass
            # Add truncation metadata as a model attribute if supported
            if hasattr(output, "model_dump"):
                data = output.model_dump()
                metadata = data.setdefault("_truncation_metadata", {})
                self._populate_truncation_metadata(data, metadata, context)
                # Reconstruct the model with metadata if possible
                try:
                    output = output.__class__(**data)
                except Exception:  # noqa: S110 - best-effort model reconstruction
                    pass

        return output

    def _populate_truncation_metadata(
        self,
        output: dict[str, Any],
        metadata: dict[str, Any],
        context: MiddlewareContext,
    ) -> None:
        """Populate truncation metadata on a dict output.

        Args:
            output: Output dict
            metadata: Metadata dict to populate
            context: Middleware context
        """
        # Check if already populated
        if metadata.get("recorded"):
            return

        limits = self._get_output_limits(context)
        output_bytes = len(str(output).encode("utf-8"))
        output_lines = str(output).count("\n") + 1

        truncated = False
        stream = None
        bytes_before = output_bytes
        lines_before = output_lines
        action = None

        if output_bytes > limits.max_stdout_bytes:
            truncated = True
            stream = "stdout"
            bytes_before = limits.max_stdout_bytes
            lines_before = min(output_lines, limits.max_lines or output_lines)
            action = "clipped"
        elif output_lines > (limits.max_lines or output_lines):
            truncated = True
            stream = "stdout"
            lines_before = limits.max_lines
            action = "dropped"

        metadata.update(
            {
                "recorded": True,
                "truncated": truncated,
                "stream": stream,
                "bytes_before": bytes_before,
                "lines_before": lines_before,
                "action": action,
                "max_bytes": limits.max_stdout_bytes,
                "max_lines": limits.max_lines,
            }
        )

    def _get_output_limits(self, context: MiddlewareContext) -> Any:
        """Get output limits from config.

        Args:
            context: Middleware context

        Returns:
            OutputLimits with configured limits
        """
        from local_coding_assistant.tools_platform.supervisor import OutputLimits

        return OutputLimits(
            max_stdout_bytes=262144,
            max_stderr_bytes=262144,
            max_lines=400,
        )

    def _get_backend_type(self) -> BackendType:
        """Return the configured backend type as a canonical enum.

        Returns:
            Backend type enum

        Raises:
            ToolBackendError: If backend returns invalid type
        """
        try:
            return BackendType(self.backend.get_backend_type())
        except (TypeError, ValueError) as exc:
            raise ToolBackendError(
                "Configured backend returned an invalid backend type",
                details={"backend_type": self.backend.get_backend_type()},
                causal_error_type=type(exc).__name__,
            ) from exc
