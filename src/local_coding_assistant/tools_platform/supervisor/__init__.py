"""Supervisor package for execution supervision.

This package provides shared execution supervision infrastructure with
backend-specific process transport support.

The main entry point is ProcessSupervisor, which uses a ProcessTransport
implementation (e.g., HostProcessTransport) for backend-specific process operations.
"""

from local_coding_assistant.tools_platform.supervisor.host_transport import (
    HostProcessTransport,
)
from local_coding_assistant.tools_platform.supervisor.supervisor import (
    ExecutionConfig,
    OutputLimits,
    ProcessSupervisor,
    TimeoutPolicy,
)
from local_coding_assistant.tools_platform.supervisor.transport import (
    ProcessTransport,
    TransportConfig,
)

__all__ = [
    "ExecutionConfig",
    "HostProcessTransport",
    "OutputLimits",
    "ProcessSupervisor",
    "ProcessTransport",
    "TimeoutPolicy",
    "TransportConfig",
]
