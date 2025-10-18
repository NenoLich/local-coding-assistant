import pytest

from local_coding_assistant.core.exceptions import (
    AgentError,
    CLIError,
    ConfigError,
    LLMError,
    LocalAssistantError,
    RuntimeFlowError,
    ToolRegistryError,
)


def test_local_assistant_error_default_subsystem_prefix():
    err = LocalAssistantError("something happened")
    assert str(err) == "[core] something happened"


def test_local_assistant_error_override_subsystem():
    err = LocalAssistantError("boom", subsystem="custom")
    assert str(err) == "[custom] boom"


@pytest.mark.parametrize(
    "exc_class,expected",
    [
        (AgentError, "[agent] oops"),
        (LLMError, "[llm] oops"),
        (ToolRegistryError, "[tools] oops"),
        (RuntimeFlowError, "[runtime] oops"),
        (ConfigError, "[config] oops"),
        (CLIError, "[cli] oops"),
    ],
)
def test_subsystem_specific_errors(exc_class, expected):
    err = exc_class("oops")
    assert str(err) == expected
