from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ExecutionMode(str, Enum):
    """Supported execution modes for prompt composition."""

    REASONING_ONLY = "reasoning_only"
    CLASSIC_TOOLS = "classic_tools"
    SANDBOX_PYTHON = "sandbox_python"
    SANDBOX_SHELL = "sandbox_shell"

    @property
    def template_name(self) -> str:
        """Return the template file associated with this execution mode."""
        if self is ExecutionMode.SANDBOX_PYTHON:
            return "modes/sandbox_python.jinja2"
        if self is ExecutionMode.SANDBOX_SHELL:
            return "modes/sandbox_shell.jinja2"
        if self is ExecutionMode.CLASSIC_TOOLS:
            return "modes/classic_tools.jinja2"
        return "modes/reasoning_only.jinja2"


class AgentProfile(BaseModel):
    """Describes how an agent should speak, reason and operate."""

    name: str
    kind: str = "default"
    description: str
    goals: list[str] = Field(default_factory=list)
    tone: str | None = None
    constraints: list[str] = Field(default_factory=list)

    @classmethod
    def default(cls) -> AgentProfile:
        return cls(
            name="default",
            kind="default",
            description=(
                "Primary coding assistant focused on safe, step-by-step reasoning "
                "with practical guidance."
            ),
            goals=[
                "Deliver concise answers grounded in repository state",
                "Surface trade-offs and assumptions explicitly",
            ],
            tone="Confident, pragmatic, collaborative",
            constraints=[
                "Never fabricate file paths or code",
                "Prefer actionable steps over vague suggestions",
            ],
        )

    @classmethod
    def planner(cls) -> AgentProfile:
        return cls(
            name="planner",
            kind="planner",
            description=(
                "Decomposes the request into executable steps and highlights risks."
            ),
            goals=[
                "Summarize objectives",
                "Outline numbered plan with verification points",
            ],
            tone="Analytical and structured",
        )

    @classmethod
    def executor(cls) -> AgentProfile:
        return cls(
            name="executor",
            kind="executor",
            description="Executes the plan, writes code, and validates results.",
            goals=[
                "Apply plan precisely",
                "Capture diffs and side-effects",
                "Report blockers or verifications needed",
            ],
            tone="Hands-on and detail oriented",
        )


class PromptContext(BaseModel):
    """Contextual payload that will eventually feed the prompt templates."""

    session_id: str
    execution_mode: ExecutionMode
    tool_call_mode: str
    user_input: str
    agent_profiles: list[AgentProfile] = Field(default_factory=list)
    active_skills: list[str] = Field(default_factory=list)
    memories: list[str] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    history: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    product_name: str = "Local Coding Assistant"
    is_sandbox_enabled: bool = False
    examples: list[dict[str, str]] = Field(default_factory=list)


class RenderedPrompt(BaseModel):
    """Materialized prompt ready for LLM consumption."""

    system_messages: list[str] = Field(default_factory=list)
    user_messages: list[str] = Field(default_factory=list)
    tool_schemas: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "AgentProfile",
    "ExecutionMode",
    "PromptContext",
    "RenderedPrompt",
]
