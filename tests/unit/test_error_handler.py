import io
import logging

from local_coding_assistant.core.error_handler import handle_error, safe_entrypoint
from local_coding_assistant.core.exceptions import CLIError
from local_coding_assistant.utils.logging import setup_logging


def test_handle_error_logs_known_local_assistant_error_as_error():
    buf = io.StringIO()
    setup_logging(level=logging.DEBUG, stream=buf)

    err = CLIError("invalid flag")
    handle_error(err)

    out = buf.getvalue()
    # ERROR emoji and level
    assert "🔥 [ERROR" in out
    # Formatted message from exceptions base
    assert "[cli] invalid flag" in out


def test_handle_error_logs_unknown_exception_as_critical():
    buf = io.StringIO()
    setup_logging(level=logging.DEBUG, stream=buf)

    err = ValueError("boom")
    handle_error(err)

    out = buf.getvalue()
    # CRITICAL emoji and level
    assert "💀 [CRITICAL" in out
    # Includes class name and message
    assert "Unexpected error: ValueError: boom" in out


def test_handle_error_verbose_includes_traceback_in_debug():
    buf = io.StringIO()
    setup_logging(level=logging.DEBUG, stream=buf)

    try:
        raise RuntimeError("trace-me")
    except RuntimeError as e:
        handle_error(e, context="unit", verbose=True)

    out = buf.getvalue()
    # DEBUG messages are included at level DEBUG
    assert "Traceback:\n" in out
    assert "RuntimeError: trace-me" in out
    # Context prefix should be present in the critical line
    assert "[unit] Unexpected error: RuntimeError: trace-me" in out


def test_safe_entrypoint_returns_function_result_and_passes_kwargs():
    buf = io.StringIO()
    setup_logging(level=logging.DEBUG, stream=buf)

    @safe_entrypoint("unit.ok")
    def f(x: int, *, verbose: bool = False) -> int:
        return x + 1

    assert f(41, verbose=False) == 42
    # Should not log errors
    assert buf.getvalue() == ""


def test_safe_entrypoint_catches_exception_logs_and_returns_none():
    buf = io.StringIO()
    setup_logging(level=logging.DEBUG, stream=buf)

    @safe_entrypoint("unit.fail")
    def g(*, verbose: bool = True):
        raise CLIError("bad input")

    res = g(verbose=True)
    assert res is None

    out = buf.getvalue()
    # Should log as ERROR (known LocalAssistantError)
    assert "🔥 [ERROR" in out
    assert "[unit.fail] [cli] bad input" in out
