"""Middleware pipeline for tool execution.

This submodule provides composable middleware components for the
tool execution pipeline. Middleware are executed in order per phase:

PREPARE phase (sequential):
1. Exposure - filter tools visible to LLM
2. Validation - validate JSON shape, normalize paths
3. Authorization - map effects to permissions, evaluate approval
4. Workspace admission - check path scope, transaction state, conflicts

EXECUTE phase (sequential):
5. Scheduling - classify effects, acquire locks
6. Execution - select backend, supervise streams, capture metrics

FINALIZE phase (sequential):
7. Normalization - convert returns/exceptions to canonical result
"""

from local_coding_assistant.tools_platform.middleware.base import (
    ExecuteMiddleware,
    FinalizeMiddleware,
    MiddlewareContext,
    MiddlewarePipeline,
    MiddlewareStage,
    PrepareMiddleware,
)
from local_coding_assistant.tools_platform.middleware.execute import (
    ExecutionMiddleware,
    SchedulingMiddleware,
)
from local_coding_assistant.tools_platform.middleware.finalize import (
    NormalizationMiddleware,
)
from local_coding_assistant.tools_platform.middleware.prepare import (
    AuthorizationMiddleware,
    ExposureMiddleware,
    ValidationMiddleware,
    WorkspaceAdmissionMiddleware,
)

__all__ = [
    "AuthorizationMiddleware",
    "ExecuteMiddleware",
    "ExecutionMiddleware",
    "ExposureMiddleware",
    "FinalizeMiddleware",
    "MiddlewareContext",
    "MiddlewarePipeline",
    "MiddlewareStage",
    "NormalizationMiddleware",
    "PrepareMiddleware",
    "SchedulingMiddleware",
    "ValidationMiddleware",
    "WorkspaceAdmissionMiddleware",
]
