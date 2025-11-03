class LocalAssistantError(Exception):
    """Base exception for all Local Assistant errors.

    The message is automatically prefixed with the subsystem name in square brackets.
    """

    subsystem = "core"

    def __init__(self, message: str, *, subsystem: str | None = None) -> None:
        self.subsystem = subsystem or self.subsystem
        super().__init__(f"[{self.subsystem}] {message}")


# ─── Subsystem-level exceptions ───────────────────────────────────────────────


class AgentError(LocalAssistantError):
    """Raised for issues related to the LLM/agent subsystem."""

    subsystem = "agent"


class LLMError(AgentError):
    """Raised for issues specific to LLM communication or generation.

    This class combines the core LLM error handling with provider-specific
    error handling.
    """

    subsystem = "llm"


class ToolRegistryError(LocalAssistantError):
    """Raised for tool registration or execution errors."""

    subsystem = "tools"


class RuntimeFlowError(LocalAssistantError):
    """Raised for runtime orchestration and task management issues."""

    subsystem = "runtime"


class ConfigError(LocalAssistantError):
    """Raised for configuration loading or parsing errors."""

    subsystem = "config"


class CLIError(LocalAssistantError):
    """Raised for CLI-specific logic or user input issues."""

    subsystem = "cli"
