"""Canonical typed contracts for tool execution platform.

This module defines the core data models for the tools platform:
- ToolDefinition: stable tool metadata and schema
- ToolCallRequest: incoming tool call from LLM
- ValidatedToolCall: parsed and normalized call
- ToolDecision: approval/deny/ask outcome
- ToolExecutionResult: canonical execution result
- ToolExecutionError: stable error contract
- ToolOutputChunk: streaming output
- Workspace models: ChangeSet, FileObservation, FileSnapshot, ConflictReport, TransactionState
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from local_coding_assistant.tools_platform.errors import (
    ToolErrorCategory,
    ToolErrorCode,
)


class OperationEffect(str, Enum):
    """Operation effect for concurrency classification."""

    PURE = "pure"
    READ = "read"
    WRITE = "write"
    PROCESS = "process"
    NETWORK = "network"
    INDEX_MUTATION = "index_mutation"


class LockMode(str, Enum):
    """Lock mode for concurrency control."""

    SHARED = "shared"
    EXCLUSIVE = "exclusive"
    NONE = "none"


class ConflictRule(str, Enum):
    """How to handle conflicts."""

    WAIT = "wait"
    REJECT = "reject"
    REQUIRE_FRESH = "require_fresh"


class ToolDecisionOutcome(str, Enum):
    """Approval decision outcomes."""

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


class ApprovalResponse(str, Enum):
    """User approval response options.

    YES_ONCE: Run this exact command once.
    YES_ALWAYS_SESSION: Whitelist this exact tool/command prefix for the remainder of the session (in-memory).
    YES_ALWAYS_PROJECT: Whitelist this exact tool/command prefix for the project (persistent).
    NO: Reject.
    """

    YES_ONCE = "yes_once"
    YES_ALWAYS_SESSION = "yes_always_session"
    YES_ALWAYS_PROJECT = "yes_always_project"
    NO = "no"


class LockLifetime(str, Enum):
    """Lifetime of a lock."""

    VALIDATION = "validation"
    EXECUTION = "execution"
    COMMIT = "commit"
    DETACHED_PROCESS = "detached_process"


def file_resource_key(path: str) -> str:
    """Generate a file resource key for locking.

    Args:
        path: File path relative to workspace

    Returns:
        Resource key string (e.g., "file:src/main.py")
    """
    return f"file:{path}"


def directory_resource_key(path: str) -> str:
    """Generate a directory resource key for locking.

    Args:
        path: Directory path relative to workspace

    Returns:
        Resource key string (e.g., "dir:src")
    """
    return f"dir:{path}"


def process_resource_key(handle_id: str) -> str:
    """Generate a process resource key for locking.

    Args:
        handle_id: Process handle ID

    Returns:
        Resource key string (e.g., "process:abc123")
    """
    return f"process:{handle_id}"


def compute_file_read_locks(arguments: dict[str, Any]) -> list[ResourceLock]:
    """Compute resource locks for file read operations.

    Args:
        arguments: Tool arguments containing 'path'

    Returns:
        List of ResourceLock (file:{path} with SHARED)
    """
    path = arguments.get("path", "")
    return [
        ResourceLock(key=file_resource_key(path), mode=LockMode.SHARED),
    ]


def compute_file_write_locks(arguments: dict[str, Any]) -> list[ResourceLock]:
    """Compute resource locks for file write operations.

    Args:
        arguments: Tool arguments containing 'path'

    Returns:
        List of ResourceLock (file:{path} with EXCLUSIVE)
    """
    path = arguments.get("path", "")
    return [
        ResourceLock(key=file_resource_key(path), mode=LockMode.EXCLUSIVE),
    ]


def compute_file_create_locks(arguments: dict[str, Any]) -> list[ResourceLock]:
    """Compute resource locks for file creation operations.

    Args:
        arguments: Tool arguments containing 'path'

    Returns:
        List of ResourceLock (file:{path} EXCLUSIVE, dir:{parent} EXCLUSIVE)
    """
    from pathlib import Path

    path = arguments.get("path", "")
    parent_dir = str(Path(path).parent) if path else ""
    locks = [ResourceLock(key=file_resource_key(path), mode=LockMode.EXCLUSIVE)]
    if parent_dir and parent_dir != ".":
        locks.append(
            ResourceLock(
                key=directory_resource_key(parent_dir), mode=LockMode.EXCLUSIVE
            )
        )
    return locks


def compute_file_delete_locks(arguments: dict[str, Any]) -> list[ResourceLock]:
    """Compute resource locks for file deletion operations.

    Args:
        arguments: Tool arguments containing 'path'

    Returns:
        List of ResourceLock (file:{path} EXCLUSIVE, dir:{parent} EXCLUSIVE)
    """
    from pathlib import Path

    path = arguments.get("path", "")
    parent_dir = str(Path(path).parent) if path else ""
    locks = [ResourceLock(key=file_resource_key(path), mode=LockMode.EXCLUSIVE)]
    if parent_dir and parent_dir != ".":
        locks.append(
            ResourceLock(
                key=directory_resource_key(parent_dir), mode=LockMode.EXCLUSIVE
            )
        )
    return locks


def compute_command_locks(arguments: dict[str, Any]) -> list[ResourceLock]:
    """Compute resource locks for command execution.

    Args:
        arguments: Tool arguments (handle_id will be added after execution starts)

    Returns:
        List of ResourceLock (processes EXCLUSIVE, conditional workspace lock)
    """
    mutation_risk = arguments.get("mutation_risk", "none")
    locks = [ResourceLock(key="processes", mode=LockMode.EXCLUSIVE)]

    # Only lock workspace exclusively for high-risk mutations
    if mutation_risk == "global":
        locks.append(ResourceLock(key="workspace", mode=LockMode.EXCLUSIVE))
    else:
        # For safe commands, only need shared workspace access
        locks.append(ResourceLock(key="workspace", mode=LockMode.SHARED))

    return locks


def compute_process_status_locks(arguments: dict[str, Any]) -> list[ResourceLock]:
    """Compute resource locks for process status checks.

    Args:
        arguments: Tool arguments containing 'handle_id'

    Returns:
        List of ResourceLock (process:{handle_id} SHARED)
    """
    handle_id = arguments.get("handle_id", "")
    return [ResourceLock(key=process_resource_key(handle_id), mode=LockMode.SHARED)]


def compute_process_kill_locks(arguments: dict[str, Any]) -> list[ResourceLock]:
    """Compute resource locks for process termination.

    Args:
        arguments: Tool arguments containing 'handle_id'

    Returns:
        List of ResourceLock (process:{handle_id} EXCLUSIVE)
    """
    handle_id = arguments.get("handle_id", "")
    return [ResourceLock(key=process_resource_key(handle_id), mode=LockMode.EXCLUSIVE)]


class BackendType(str, Enum):
    """Available execution backends."""

    HOST = "host"
    DOCKER = "docker"
    REMOTE = "remote"


class ToolRiskClass(str, Enum):
    """Risk classification for tools."""

    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ToolEventType(str, Enum):
    """Event types emitted during tool execution."""

    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_COMPLETE = "tool_call_complete"
    TOOL_CALL_ERROR = "tool_call_error"
    TOOL_CALL_DENIED = "tool_call_denied"
    TOOL_CALL_CANCELLED = "tool_call_cancelled"
    STREAM_CHUNK = "stream_chunk"
    CONFLICT_DETECTED = "conflict_detected"


@dataclass
class ToolEvent:
    """Event emitted during tool execution for dashboard/logs/telemetry.

    Attributes:
        type: Event type
        call_id: Tool call identifier
        session_id: Session identifier
        frame_id: Execution frame identifier
        trace_id: Trace identifier for correlation
        timestamp: When the event occurred
        data: Event-specific data (timing, resource usage, changeset, etc.)
    """

    type: ToolEventType
    call_id: str
    session_id: str
    frame_id: str | None
    trace_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    data: dict[str, Any] = field(default_factory=dict)


class ToolExecutionStatus(str, Enum):
    """Execution status values."""

    COMPLETED = "completed"
    FAILED = "failed"
    DENIED = "denied"
    PENDING_APPROVAL = "pending_approval"
    CANCELLED = "cancelled"


# Tool statuses an action can never transition out of. Derived from the enum
# above so consumers (resume skip checks, outcome derivation, transcript)
# cannot drift from the status contract.
TERMINAL_TOOL_STATUSES: frozenset[str] = frozenset(
    status.value
    for status in (
        ToolExecutionStatus.COMPLETED,
        ToolExecutionStatus.FAILED,
        ToolExecutionStatus.DENIED,
        ToolExecutionStatus.CANCELLED,
    )
)


@dataclass(frozen=True)
class ResourceLock:
    """A single resource lock specification.

    Attributes:
        key: Resource identifier (workspace, file:path, dir:path, etc.)
        mode: Lock mode for this resource
    """

    key: str
    mode: LockMode = LockMode.NONE


@dataclass(frozen=True)
class ConcurrencyScope:
    """Describes the resource namespace a call locks while running.

    Attributes:
        effect: Operation effect type
        resource_locks: List of ResourceLock specifications with keys and modes.
            Can be a static list or a callable that computes locks from tool arguments.
        capacity_group: Optional semaphore group name
        conflict_rule: How to handle conflicts
        lifetime: When the lock is released
    """

    effect: OperationEffect
    resource_locks: (
        list[ResourceLock] | Callable[[dict[str, Any]], list[ResourceLock]]
    ) = field(default_factory=list)
    capacity_group: str | None = None
    conflict_rule: ConflictRule = ConflictRule.WAIT
    lifetime: LockLifetime = LockLifetime.EXECUTION


@dataclass(frozen=True)
class TimeoutProfile:
    """Timeout policy for tool execution.

    Attributes:
        soft_timeout: Soft timeout before asking user
        hard_timeout: Maximum execution deadline
        idle_timeout: Timeout when no output is produced
        heartbeat_interval: Interval for progress checks
        cancellation_grace: Grace period before force kill
    """

    soft_timeout: timedelta = timedelta(seconds=10)
    hard_timeout: timedelta = timedelta(seconds=300)
    idle_timeout: timedelta = timedelta(seconds=30)
    heartbeat_interval: timedelta = timedelta(seconds=5)
    cancellation_grace: timedelta = timedelta(seconds=5)


class ToolDefinition(BaseModel):
    """Stable tool metadata and schema.

    Attributes:
        id: Stable tool identifier (e.g., "search_text")
        version: Tool version for compatibility tracking
        description: Human-readable description for LLM
        input_schema: JSON Schema for input validation
        output_schema: JSON Schema for output validation
        permissions: Required permissions
        risk_class: Risk classification
        effects: Operation effects
        supported_backends: Backends this tool can run on
        supports_streaming: Whether tool supports streamed output
        concurrency_scope: Concurrency control configuration
        timeout_profile: Timeout configuration
        exposure_policy: When this tool is exposed to LLM
    """

    id: str
    version: str = "1.0.0"
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    permissions: list[str] = Field(default_factory=list)
    risk_class: ToolRiskClass = ToolRiskClass.LOW
    effects: list[OperationEffect] = Field(default_factory=list)
    supported_backends: list[BackendType] = Field(
        default_factory=lambda: [BackendType.HOST]
    )
    supports_streaming: bool = False
    concurrency_scope: ConcurrencyScope | None = None
    timeout_profile: TimeoutProfile | None = None
    exposure_policy: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}


class ToolCallRequest(BaseModel):
    """Incoming tool call from LLM.

    Attributes:
        call_id: Unique identifier for this call
        session_id: Session identifier
        frame_id: Execution frame identifier
        tool_id: Tool to execute
        raw_arguments: Raw arguments from LLM
        requested_backend: Preferred backend (optional)
        workspace_id: Workspace identifier
        transaction_id: Transaction identifier (optional)
        turn_id: Turn identifier for checkpoint tracking (optional)
        parent_call_id: Parent call for nested tool calls
        deadline_policy: Optional deadline override
        confirmation_token: Token from previous ASK decision (for retry)
        approval_response: User's approval response (for retry)
    """

    call_id: str
    session_id: str
    frame_id: str
    tool_id: str
    raw_arguments: dict[str, Any]
    requested_backend: BackendType | None = None
    workspace_id: str
    transaction_id: str | None = None
    turn_id: str | None = None
    parent_call_id: str | None = None
    deadline_policy: timedelta | None = None
    confirmation_token: str | None = None
    approval_response: ApprovalResponse | None = None


class ValidatedToolCall(BaseModel):
    """Parsed and normalized tool call ready for execution.

    Attributes:
        request: Original request
        tool_definition: Tool definition used for validation
        input_model: Validated input model instance
        normalized_paths: Normalized file paths
        execution_metadata: Additional execution context
    """

    request: ToolCallRequest
    tool_definition: ToolDefinition
    input_model: BaseModel
    normalized_paths: dict[str, str] = Field(default_factory=dict)
    execution_metadata: dict[str, Any] = Field(default_factory=dict)


class ToolDecision(BaseModel):
    """Approval/deny/ask decision for a tool call.

    Attributes:
        outcome: Allow, deny, or ask
        reason: Human-readable explanation
        policy_rule: Rule that produced this decision
        confirmation_token: Token for user confirmation (if ask)
        expiry: When this decision expires
        decision_time: When this decision was made
        approval_response: User's approval response (if applicable)
        proposed_rule: The rule that would be created if user approves (for ASK decisions)
    """

    outcome: ToolDecisionOutcome
    reason: str
    policy_rule: str | None = None
    confirmation_token: str | None = None
    expiry: datetime | None = None
    decision_time: datetime = Field(default_factory=datetime.utcnow)
    approval_response: ApprovalResponse | None = None
    proposed_rule: Any | None = None  # PolicyRule for ASK decisions


class ToolExecutionError(BaseModel):
    """Canonical error contract for tool execution.

    Attributes:
        code: Stable error code
        category: High-level error category
        message: Message safe for the LLM
        retryable: Whether this error is retryable
        remediation: Suggested remediation steps
        details: Structured error details
        causal_error_type: Type of the original exception
    """

    code: ToolErrorCode
    category: ToolErrorCategory
    message: str
    retryable: bool = False
    remediation: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    causal_error_type: str | None = None


class ToolOutputChunk(BaseModel):
    """Streaming output chunk.

    Attributes:
        stream_id: Stream identifier
        sequence: Sequence number for ordering
        channel: Output channel (stdout, stderr, progress, status)
        payload: Text or structured data
        timestamp: When this chunk was emitted
        cumulative_bytes: Total bytes emitted so far
    """

    stream_id: str
    sequence: int
    channel: str
    payload: str | dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    cumulative_bytes: int = 0


class ChangeSet(BaseModel):
    """Set of file changes from a tool execution.

    Attributes:
        changes: List of individual file changes
        total_bytes_changed: Total bytes modified
        total_files_changed: Number of files affected
    """

    changes: list[FileChange] = Field(default_factory=list)
    total_bytes_changed: int = 0
    total_files_changed: int = 0


class FileChange(BaseModel):
    """Individual file change record.

    Attributes:
        path: Relative path to file
        before_hash: Hash before change
        after_hash: Hash after change
        before_size: Size before change
        after_size: Size after change
        before_lines: Line count before change
        after_lines: Line count after change
        change_type: Type of change (create, modify, delete)
    """

    path: str
    before_hash: str | None = None
    after_hash: str | None = None
    before_size: int | None = None
    after_size: int | None = None
    before_lines: int | None = None
    after_lines: int | None = None
    change_type: str = "modify"


class ObservationSource(str, Enum):
    """Source of a file observation."""

    READ = "read"
    WRITE = "write"
    SNAPSHOT = "snapshot"
    AGENT_CREATE = "agent_create"
    EXTERNAL = "external"


class FileObservation(BaseModel):
    """Observed file state at a point in time.

    Attributes:
        path: Relative path to file
        hash: Content hash
        size: File size in bytes
        mtime: Modification time
        inode: File identifier (platform-specific)
        observed_at: When this observation was made
        source: Origin of this observation (read, write, snapshot, agent_create, external)
    """

    path: str
    hash: str
    size: int
    mtime: datetime
    inode: str | None = None
    observed_at: datetime = Field(default_factory=datetime.utcnow)
    source: ObservationSource = ObservationSource.READ


class FileSnapshot(BaseModel):
    """Snapshot of file content.

    Attributes:
        path: Relative path to file
        content: File content
        observation: File observation metadata
        snapshot_id: Snapshot identifier
    """

    path: str
    content: str
    observation: FileObservation
    snapshot_id: str


class ConflictReport(BaseModel):
    """Report of a write conflict.

    Attributes:
        path: File path with conflict
        base_hash: Hash at transaction start
        current_hash: Hash when conflict detected
        proposed_hash: Hash of proposed change
        conflict_type: Type of conflict
        conflict_detected: Whether a conflict was actually detected
        resolution_options: Available resolution choices
        detected_at: When conflict was detected
        external_diff: Diff showing what changed on disk since baseline (conflict only)
        proposed_diff: Diff showing what the proposed edit would do (conflict only)
        recommendation: Recommended action: re_read_file or retry_or_adapt (conflict only)
        message: Human-readable explanation for the LLM (conflict only)
    """

    path: str
    base_hash: str
    current_hash: str
    proposed_hash: str
    conflict_type: str
    conflict_detected: bool = False
    resolution_options: list[str] = Field(default_factory=list)
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    external_diff: str | None = None
    proposed_diff: str | None = None
    recommendation: str | None = None
    message: str | None = None


class TransactionState(str, Enum):
    """States of a workspace transaction."""

    ACTIVE = "active"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    CONFLICTED = "conflicted"


class ProcessState(str, Enum):
    """States of a process managed by LOCCA."""

    SPAWNING = "spawning"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class ProcessHandle(BaseModel):
    """Opaque handle for a LOCCA-managed process.

    Wraps the OS PID with internal state and metadata. The handle is
    session-scoped and expires when the backend/session is torn down.

    Attributes:
        handle_id: Opaque session-scoped identifier
        os_pid: OS process ID (for diagnostics only, not for control)
        state: Current process state
        backend: Backend that owns this process
        start_time: When the process was spawned
        last_activity: Last time output was produced
        return_code: Return code if completed
        exit_signal: Signal that terminated the process (if applicable)
        resource_usage: Optional resource usage metrics
        metadata: Additional process metadata
    """

    handle_id: str
    os_pid: int | None = None
    state: ProcessState = ProcessState.SPAWNING
    backend: BackendType
    start_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_activity: datetime | None = None
    return_code: int | None = None
    exit_signal: str | None = None
    resource_usage: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolExecutionResult[T](BaseModel):
    """Canonical execution result.

    Attributes:
        status: Execution status
        output: Typed output
        error: Error if execution failed
        timing: Execution timing information
        output_chunks: Streaming output chunks
        warnings: Warning messages
        changeset: File changes made
        backend: Backend used for execution
        trace_id: Trace identifier
        decision: Approval decision used
    """

    status: str
    output: T | None = None
    error: ToolExecutionError | None = None
    timing: dict[str, float] = Field(default_factory=dict)
    output_chunks: list[ToolOutputChunk] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    changeset: ChangeSet | None = None
    backend: BackendType | None = None
    trace_id: str | None = None
    decision: ToolDecision | None = None
