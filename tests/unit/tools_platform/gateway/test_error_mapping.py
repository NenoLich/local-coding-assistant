"""Error mapping tests for tools_platform contracts.

Tests that exceptions from backends and tools are correctly converted
to the canonical ToolExecutionError contract at the gateway boundary.
"""

from pydantic import BaseModel, ValidationError

from local_coding_assistant.tools_platform.contracts import ToolExecutionError
from local_coding_assistant.tools_platform.error_mapping import map_exception
from local_coding_assistant.tools_platform.errors import (
    ToolErrorCategory,
    ToolErrorCode,
)
from local_coding_assistant.tools_platform.exceptions import (
    ToolBackendError,
    ToolValidationError,
)


class TestErrorMapping:
    """Test error mapping from exceptions to ToolExecutionError."""

    def test_validation_error_mapping(self):
        """Test validation errors map to INVALID_ARGUMENTS."""
        # Simulate a validation error from a tool
        try:
            raise ValueError("Invalid argument: missing required field")
        except ValueError as e:
            error = ToolExecutionError(
                code=ToolErrorCode.INVALID_ARGUMENTS,
                category=ToolErrorCategory.VALIDATION,
                message=str(e),
                causal_error_type=type(e).__name__,
            )

        assert error.code == ToolErrorCode.INVALID_ARGUMENTS
        assert error.category == ToolErrorCategory.VALIDATION
        assert error.causal_error_type == "ValueError"
        assert "missing required field" in error.message

    def test_validation_error_strips_pydantic_url(self):
        """Pydantic validation errors should not surface the errors.pydantic.dev URL."""

        class M(BaseModel):
            includes: list[str]

        try:
            M(includes="*.py")
        except ValidationError as exc:
            error = map_exception(exc, tool_id="search_text")

        assert error.code == ToolErrorCode.INVALID_ARGUMENTS
        assert error.category == ToolErrorCategory.VALIDATION
        assert "errors.pydantic.dev" not in error.message
        assert "Input should be a valid list" in error.message

    def test_permission_error_mapping(self):
        """Test permission errors map to DENIED."""
        # Simulate a permission error
        try:
            raise PermissionError("Access denied to file")
        except PermissionError as e:
            error = ToolExecutionError(
                code=ToolErrorCode.DENIED,
                category=ToolErrorCategory.AUTHORIZATION,
                message="Permission denied",
                causal_error_type=type(e).__name__,
            )

        assert error.code == ToolErrorCode.DENIED
        assert error.category == ToolErrorCategory.AUTHORIZATION

    def test_file_not_found_error_mapping(self):
        """Test file not found errors map to INVALID_ARGUMENTS."""
        try:
            raise FileNotFoundError("File not found: test.py")
        except FileNotFoundError as e:
            error = ToolExecutionError(
                code=ToolErrorCode.INVALID_ARGUMENTS,
                category=ToolErrorCategory.VALIDATION,
                message="File not found",
                causal_error_type=type(e).__name__,
            )

        assert error.code == ToolErrorCode.INVALID_ARGUMENTS

    def test_timeout_error_mapping(self):
        """Test timeout errors map to TIMEOUT."""
        try:
            raise TimeoutError("Operation timed out")
        except TimeoutError as e:
            error = ToolExecutionError(
                code=ToolErrorCode.TIMEOUT,
                category=ToolErrorCategory.RESOURCE,
                message="Operation timed out",
                retryable=True,
                causal_error_type=type(e).__name__,
            )

        assert error.code == ToolErrorCode.TIMEOUT
        assert error.category == ToolErrorCategory.RESOURCE
        assert error.retryable is True

    def test_connection_error_mapping(self):
        """Test connection errors map to BACKEND_FAILURE."""
        try:
            raise ConnectionError("Failed to connect to backend")
        except ConnectionError as e:
            error = ToolExecutionError(
                code=ToolErrorCode.BACKEND_FAILURE,
                category=ToolErrorCategory.INFRASTRUCTURE,
                message="Backend connection failed",
                retryable=True,
                causal_error_type=type(e).__name__,
            )

        assert error.code == ToolErrorCode.BACKEND_FAILURE
        assert error.category == ToolErrorCategory.INFRASTRUCTURE
        assert error.retryable is True

    def test_generic_exception_mapping(self):
        """Test generic exceptions map to INTERNAL_FAILURE."""
        try:
            raise RuntimeError("Unexpected error")
        except RuntimeError as e:
            error = ToolExecutionError(
                code=ToolErrorCode.INTERNAL_FAILURE,
                category=ToolErrorCategory.UNKNOWN,
                message="An internal error occurred",
                causal_error_type=type(e).__name__,
            )

        assert error.code == ToolErrorCode.INTERNAL_FAILURE
        assert error.category == ToolErrorCategory.UNKNOWN

    def test_error_with_remediation(self):
        """Test error with remediation information."""
        error = ToolExecutionError(
            code=ToolErrorCode.PATH_VIOLATION,
            category=ToolErrorCategory.VALIDATION,
            message="Path outside workspace",
            remediation="Ensure the path is within the workspace root",
            details={"requested_path": "/etc/passwd", "workspace_root": "/project"},
        )

        assert error.remediation is not None
        assert "workspace root" in error.remediation.lower()
        assert error.details["requested_path"] == "/etc/passwd"

    def test_error_retryable_flag(self):
        """Test retryable flag is set correctly for different errors."""
        retryable_errors = [
            (ToolErrorCode.TIMEOUT, True),
            (ToolErrorCode.BACKEND_FAILURE, True),
            (ToolErrorCode.INVALID_ARGUMENTS, False),
            (ToolErrorCode.DENIED, False),
            (ToolErrorCode.INTERNAL_FAILURE, False),
        ]

        for code, expected_retryable in retryable_errors:
            error = ToolExecutionError(
                code=code,
                category=ToolErrorCategory.UNKNOWN,
                message="Test",
                retryable=expected_retryable,
            )
            assert error.retryable == expected_retryable

    def test_error_serialization_preserves_all_fields(self):
        """Test that error serialization preserves all fields."""
        error = ToolExecutionError(
            code=ToolErrorCode.CONFLICT,
            category=ToolErrorCategory.CONCURRENCY,
            message="Write conflict detected",
            retryable=False,
            remediation="Refresh and retry",
            details={"conflict_type": "concurrent_write"},
            causal_error_type="ConflictError",
        )

        data = error.model_dump()
        restored = ToolExecutionError.model_validate(data)

        assert restored.code == ToolErrorCode.CONFLICT
        assert restored.category == ToolErrorCategory.CONCURRENCY
        assert restored.retryable is False
        assert restored.remediation == "Refresh and retry"
        assert restored.details["conflict_type"] == "concurrent_write"
        assert restored.causal_error_type == "ConflictError"

    def test_error_message_safe_for_llm(self):
        """Test that error messages are safe for LLM consumption."""
        # Messages should not contain stack traces or internal paths
        safe_messages = [
            "Tool execution failed",
            "Invalid input provided",
            "Permission denied",
            "Resource not available",
        ]

        for msg in safe_messages:
            error = ToolExecutionError(
                code=ToolErrorCode.INTERNAL_FAILURE,
                category=ToolErrorCategory.UNKNOWN,
                message=msg,
            )
            # Should not contain internal details
            assert "traceback" not in error.message.lower()
            assert "stack trace" not in error.message.lower()
            assert "/" not in error.message or "path" in error.message.lower()

    def test_platform_exception_mapping(self):
        """Test platform-specific exceptions are converted to ToolExecutionError."""
        platform_error = ToolValidationError(
            "Missing required field",
            tool_id="read_file",
            call_id="call-123",
            workspace_id="ws-1",
        )

        error = map_exception(
            platform_error,
            tool_id="read_file",
            call_id="call-999",
        )

        assert error.code == ToolErrorCode.INVALID_ARGUMENTS
        assert error.category == ToolErrorCategory.VALIDATION
        assert error.details["tool_id"] == "read_file"
        assert error.details["call_id"] == "call-999"
        assert "Missing required field" in error.message

    def test_backend_exception_mapping_sets_retryable_flag(self):
        """Test backend failures map to retryable infra errors."""
        backend_error = ToolBackendError(
            "Docker backend unreachable",
            backend="docker",
        )

        error = map_exception(backend_error, tool_id="run_command")

        assert error.code == ToolErrorCode.BACKEND_FAILURE
        assert error.category == ToolErrorCategory.INFRASTRUCTURE
        assert error.retryable is True
        assert error.details["tool_id"] == "run_command"
        assert error.details["backend"] == "docker"

    def test_error_details_can_contain_internal_info(self):
        """Test that error details can contain internal information for debugging."""
        error = ToolExecutionError(
            code=ToolErrorCode.INTERNAL_FAILURE,
            category=ToolErrorCategory.UNKNOWN,
            message="Internal error",
            details={
                "stack_trace": "Line 1\nLine 2\nLine 3",
                "internal_state": {"key": "value"},
                "debug_info": "Additional debug information",
            },
        )

        # Details can contain internal information
        assert "stack_trace" in error.details
        assert len(error.details["stack_trace"]) > 0

        # But message should remain safe
        assert error.message == "Internal error"
