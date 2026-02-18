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
    """Raised for issues specific to LLM communication or generation."""

    subsystem = "llm"


class LLMInfrastructureError(LLMError):
    """Raised for provider-level infrastructure issues (auth, network, rate limits)."""


class LLMContentError(LLMError):
    """Raised when the LLM returns malformed or unparseable content."""

    def __init__(self, message: str, raw_response: str | None = None, **kwargs):
        super().__init__(message, **kwargs)
        self.raw_response = raw_response


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
