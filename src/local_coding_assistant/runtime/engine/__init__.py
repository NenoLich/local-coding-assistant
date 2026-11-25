"""Execution engine and frame management."""

from local_coding_assistant.runtime.engine.approval import (
    ApprovalRequest,
    ApprovalResponse,
    ApprovalStatus,
)
from local_coding_assistant.runtime.engine.config import RuntimeLimitsConfig
from local_coding_assistant.runtime.engine.continuation import (
    FrameContinuation,
    ToolObservation,
)
from local_coding_assistant.runtime.engine.engine import IExecutionEngine, RuntimeEngine
from local_coding_assistant.runtime.engine.frame import (
    ActionKind,
    ActionRecord,
    ExecutionResult,
    ExecutionStatus,
    FrameOutcome,
    FrameOutcomeDetail,
    RuntimeFrame,
    ToolCallInfo,
    build_frame,
)
from local_coding_assistant.runtime.engine.frame_agent import FrameAgent
from local_coding_assistant.runtime.engine.limits import (
    DeadlineTracker,
    LimitCheckResult,
    LimitViolationReason,
    LoopGuard,
    RuntimeLimits,
    TokenBudget,
    ToolRetryGuard,
)
from local_coding_assistant.runtime.engine.policy import (
    ContinuationAction,
    ContinuationDecision,
    ContinuationPolicy,
    StopReason,
)

__all__ = [
    "ActionKind",
    "ActionRecord",
    "ApprovalRequest",
    "ApprovalResponse",
    "ApprovalStatus",
    "ContinuationAction",
    "ContinuationDecision",
    "ContinuationPolicy",
    "DeadlineTracker",
    "ExecutionResult",
    "ExecutionStatus",
    "FrameAgent",
    "FrameContinuation",
    "FrameOutcome",
    "FrameOutcomeDetail",
    "IExecutionEngine",
    "LimitCheckResult",
    "LimitViolationReason",
    "LoopGuard",
    "RuntimeEngine",
    "RuntimeFrame",
    "RuntimeLimits",
    "RuntimeLimitsConfig",
    "StopReason",
    "TokenBudget",
    "ToolCallInfo",
    "ToolObservation",
    "ToolRetryGuard",
    "build_frame",
]
