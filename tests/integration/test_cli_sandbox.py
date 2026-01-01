"""Integration tests for the sandbox CLI commands."""

import logging

from local_coding_assistant.cli.commands import sandbox as sandbox_cli


def test_cli_sandbox_run_executes_code(cli_runner, sandbox_cli_test_env):
    """`sandbox run` should execute provided inline code."""
    result = cli_runner.invoke(sandbox_cli.app, ["run", "print('hi')"])

    assert result.exit_code == 0, result.output
    assert "Executing code in sandbox" in result.output

    assert len(sandbox_cli_test_env.sandbox.executed_requests) == 1
    request = sandbox_cli_test_env.sandbox.executed_requests[0]
    assert request.code == "print('hi')"
    assert request.session_id == "default"
    assert request.persistence is True
    assert request.env_vars == {}
    assert sandbox_cli_test_env.bootstrap_levels[-1] == logging.INFO


def test_cli_sandbox_run_with_file_input(cli_runner, sandbox_cli_test_env, tmp_path):
    """`sandbox run --file` should load code from disk."""
    code_file = tmp_path / "snippet.py"
    code_file.write_text("print('from file')", encoding="utf-8")

    result = cli_runner.invoke(
        sandbox_cli.app,
        ["run", "--file", str(code_file)],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    request = sandbox_cli_test_env.sandbox.executed_requests[-1]
    assert request.code == code_file.read_text(encoding="utf-8")


def test_cli_sandbox_run_with_env_and_session(cli_runner, sandbox_cli_test_env):
    """`sandbox run` should forward env vars, session, timeout, and log level."""
    result = cli_runner.invoke(
        sandbox_cli.app,
        [
            "run",
            "print('env test')",
            "--env",
            "API_KEY=secret",
            "--env",
            "MODE=test",
            "--session",
            "custom-session",
            "--timeout",
            "42",
            "--log-level",
            "DEBUG",
        ],
    )

    assert result.exit_code == 0, result.output
    request = sandbox_cli_test_env.sandbox.executed_requests[-1]
    assert request.session_id == "custom-session"
    assert request.timeout == 42
    assert request.env_vars == {"API_KEY": "secret", "MODE": "test"}
    assert sandbox_cli_test_env.bootstrap_levels[-1] == logging.DEBUG


def test_cli_sandbox_run_requires_code_or_file(cli_runner):
    """`sandbox run` should fail if neither code nor file is provided."""
    result = cli_runner.invoke(sandbox_cli.app, ["run"])

    assert result.exit_code == 1
    assert "Must provide either CODE argument or --file" in result.stderr


def test_cli_sandbox_exec_wraps_shell_command(cli_runner, sandbox_cli_test_env):
    """`sandbox exec` should wrap shell commands inside Python code."""
    command = "echo hello"

    result = cli_runner.invoke(
        sandbox_cli.app,
        ["exec", command],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    request = sandbox_cli_test_env.sandbox.executed_requests[-1]
    assert "cmd = 'echo hello'" in request.code


def test_cli_sandbox_stop_session(cli_runner, sandbox_cli_test_env):
    """`sandbox stop` should stop the requested session."""
    session_id = "session-123"

    result = cli_runner.invoke(
        sandbox_cli.app,
        ["stop", session_id],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert sandbox_cli_test_env.sandbox.stop_calls == [session_id]
