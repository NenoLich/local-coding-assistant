"""Tools platform - typed, backend-neutral tool execution infrastructure.

This package provides a unified tool execution platform with:
- Canonical typed contracts for tool definitions, requests, and results
- Backend-neutral execution (host, Docker sandbox, future remote)
- Approval, exposure, validation, timeout handling, and concurrency
- Workspace transactions and conflict detection
- Response normalization and trace/events
"""

from local_coding_assistant.tools_platform.backends.base import ExecutionBackend
from local_coding_assistant.tools_platform.contracts import (
    BackendType,
    ChangeSet,
    ConflictReport,
    ConflictRule,
    FileObservation,
    FileSnapshot,
    OperationEffect,
    ToolCallRequest,
    ToolDecision,
    ToolDecisionOutcome,
    ToolDefinition,
    ToolErrorCategory,
    ToolErrorCode,
    ToolExecutionError,
    ToolExecutionResult,
    ToolOutputChunk,
    TransactionState,
    ValidatedToolCall,
)
from local_coding_assistant.tools_platform.definitions import (
    ToolBuilder,
    ToolImplementation,
)
from local_coding_assistant.tools_platform.error_mapping import map_exception
from local_coding_assistant.tools_platform.exceptions import (
    ToolAuthorizationError,
    ToolBackendError,
    ToolConflictError,
    ToolInternalError,
    ToolPathViolationError,
    ToolPlatformError,
    ToolProcessError,
    ToolTimeoutError,
    ToolValidationError,
)
from local_coding_assistant.tools_platform.gateway import (
    ApprovalService,
    ToolGateway,
)
from local_coding_assistant.tools_platform.registry import ToolRegistry
from local_coding_assistant.tools_platform.workspace import WorkspaceManager

__all__ = [
    "ApprovalService",
    "BackendType",
    "ChangeSet",
    "ConflictReport",
    "ConflictRule",
    # Backend
    "ExecutionBackend",
    "FileObservation",
    "FileSnapshot",
    "OperationEffect",
    "ToolAuthorizationError",
    "ToolBackendError",
    # Definitions
    "ToolBuilder",
    "ToolCallRequest",
    "ToolConflictError",
    "ToolDecision",
    "ToolDecisionOutcome",
    # Contracts
    "ToolDefinition",
    "ToolErrorCategory",
    "ToolErrorCode",
    "ToolExecutionError",
    "ToolExecutionResult",
    # Gateway
    "ToolGateway",
    "ToolImplementation",
    "ToolInternalError",
    "ToolOutputChunk",
    "ToolPathViolationError",
    # Exceptions
    "ToolPlatformError",
    "ToolProcessError",
    "ToolRegistry",
    "ToolTimeoutError",
    "ToolValidationError",
    "TransactionState",
    "ValidatedToolCall",
    "WorkspaceManager",
    "map_exception",
]
