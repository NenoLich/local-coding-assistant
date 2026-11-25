"""Local Coding Assistant - Runtime Package.

This module provides the core runtime orchestration for executing agent queries,
managing sessions, context resolution, event streaming, and execution engines.
"""

from local_coding_assistant.runtime.events import RunReport
from local_coding_assistant.runtime.events.pipeline import (
    EventPipeline,
    EventSubscriber,
    create_default_pipeline,
)
from local_coding_assistant.runtime.options import RunOptions
from local_coding_assistant.runtime.types import (
    AgentProfile,
    AgentRequest,
    ExecutionMode,
    PromptContext,
    RenderedPrompt,
    ToolSpec,
)

__all__ = [
    "AgentProfile",
    "AgentRequest",
    "EventPipeline",
    "EventSubscriber",
    "ExecutionMode",
    "PromptContext",
    "RenderedPrompt",
    "RunOptions",
    "RunReport",
    "ToolSpec",
    "create_default_pipeline",
]
