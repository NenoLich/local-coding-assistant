"""Core runtime types shared across the runtime module."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from local_coding_assistant.runtime.session import SessionState


class ToolSpec(BaseModel):
    """Structured representation of a tool for prompt composition."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)

    def to_openai_function(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format.

        Returns:
            Dictionary in OpenAI function calling format:
            {
                "type": "function",
                "function": {
                    "name": "...",
                    "description": "...",
                    "parameters": {...}
                }
            }
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


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
    model_policy: str = "general"

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
            model_policy="general",
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
            model_policy="planning_mode",
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
            model_policy="coding_task",
        )


class PromptContext(BaseModel):
    """Contextual payload that will eventually feed the prompt templates."""

    session_id: str
    execution_mode: ExecutionMode
    tool_call_mode: str
    agent_profile: AgentProfile | None = None  # Single active agent profile
    active_skills: list[str] = Field(default_factory=list)
    memories: list[str] = Field(default_factory=list)
    tools: list[ToolSpec] = Field(
        default_factory=list
    )  # Structured tool specifications
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Model-aware template selection
    model_id: str = ""
    model_provider: str = ""  # "anthropic" | "openai" | "google" | ...
    personality: str | None = None  # "friendly" | "pragmatic" | "concise"

    # Static context (session-level)
    project_info: Any = None  # Repository project information
    repo_map_data: Any = None  # Repository map data
    examples: list[dict[str, str]] = Field(default_factory=list)
    workspace_root: str = ""
    platform: str = ""
    agents_md_content: list[tuple[str, str]] = Field(
        default_factory=list
    )  # [(path, content)] closest first
    global_rules: str = ""
    config_instructions: list[str] = Field(default_factory=list)

    # Dynamic context
    cwd: str = ""  # Omitted if the same as workspace_root
    current_date: int = 0
    is_git_repo: bool = False
    sandbox_enabled: bool = False


class RenderedPrompt(BaseModel):
    """Materialized prompt ready for LLM consumption."""

    system_messages: list[str] = Field(default_factory=list)
    user_messages: list[str] = Field(default_factory=list)
    tool_schemas: list[dict[str, Any]] = Field(default_factory=list)
    history: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_full_prompt(self) -> list[dict[str, Any]]:
        """Build complete messages list including system, history, and user messages.

        Returns:
            Complete messages list ready for LLM provider
        """
        messages: list[dict[str, Any]] = []

        # Add system messages
        if self.system_messages:
            system_prompt = "\n\n".join(self.system_messages)
            messages.append({"role": "system", "content": system_prompt})

        # Add history
        if self.history:
            for message in self.history:
                if isinstance(message, dict) and {"role", "content"} <= message.keys():
                    messages.append(
                        {"role": message["role"], "content": message["content"]}
                    )

        # Add user messages
        for user_msg in self.user_messages:
            messages.append({"role": "user", "content": user_msg})

        return messages

    def get_user_prompt(self) -> str:
        """Build user prompt string from user messages"""
        if self.user_messages:
            return "\n\n".join(self.user_messages)
        return ""


class AgentRequest(BaseModel):
    """Standardized request contract for agent execution.

    Encapsulates all per-request parameters for agent mode execution,
    separating them from global configuration defaults.
    """

    user_input: str
    session: SessionState
    agent_mode: str | None = None
    model_override: str | None = None
    temperature_override: float | None = None
    max_tokens_override: int | None = None
    tool_call_mode_override: str | None = None
    sandbox_session_override: str | None = None
    streaming: bool | None = None
    max_iterations: int | None = None


__all__ = [
    "AgentProfile",
    "AgentRequest",
    "ExecutionMode",
    "PromptContext",
    "RenderedPrompt",
    "ToolSpec",
]
