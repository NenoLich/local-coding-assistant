"""Sandbox management commands."""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

import typer

from local_coding_assistant.core.bootstrap import bootstrap
from local_coding_assistant.core.error_handler import safe_entrypoint
from local_coding_assistant.sandbox.sandbox_types import SandboxExecutionRequest
from local_coding_assistant.utils.logging import get_logger

app = typer.Typer(name="sandbox", help="Manage and interact with the sandbox")
log = get_logger("cli.sandbox")


def _parse_env(env: list[str] | None) -> dict[str, str]:
    env_vars: dict[str, str] = {}
    if env:
        for item in env:
            if "=" in item:
                k, v = item.split("=", 1)
                env_vars[k] = v
    return env_vars


async def _run_code(
    sandbox_manager: Any,
    code: str,
    session_id: str = "default",
    env_vars: dict[str, str] | None = None,
    timeout: int = 30,
    persistence: bool = False,
) -> None:
    """Execute code in the sandbox."""
    sandbox = None
    try:
        persistent_session_msg = "persistent" if persistence else "non-persistent"
        session_id_msg = f"{session_id}" if session_id else ""
        typer.echo(
            f"Executing code in sandbox with {persistent_session_msg} session {session_id_msg}"
        )

        sandbox = sandbox_manager.get_sandbox()

        request = SandboxExecutionRequest(
            code=code,
            session_id=session_id,
            timeout=timeout,
            env_vars=env_vars or {},
            persistence=persistence,
        )
        response = await sandbox.execute(request)
        typer.echo(f"Sandbox execution response: {response}")
        typer.echo("Request in sandbox executed")
        if response.stdout:
            typer.echo(response.stdout, nl=False)
        if response.stderr:
            typer.echo(response.stderr, err=True, nl=False)
        if not response.success:
            if response.error:
                typer.echo(f"\nError: {response.error}", err=True)
            sys.exit(1)
        typer.echo("\n")

    except Exception as e:
        msg = str(e)
        if (
            "Failed to connect to Docker daemon" in msg
            or "error during connect" in msg.lower()
        ):
            typer.echo(f"\nError: {msg}", err=True)
            typer.echo("\n💡 Hint: Is Docker Desktop running?", err=True)
        else:
            typer.echo(f"Error executing code: {e}", err=True)
        sys.exit(1)


@app.command()
@safe_entrypoint("cli.sandbox.run")
def run(
    code_arg: str = typer.Argument(None, help="Python code to execute", metavar="CODE"),
    code_opt: str = typer.Option(
        None, "--code", "-c", help="Deprecated: use positional argument"
    ),
    file: str = typer.Option(
        None, "--file", "-f", help="Path to Python file to execute", exists=True
    ),
    timeout: int = typer.Option(30, help="Execution timeout in seconds"),
    env: list[str] = typer.Option(  # noqa: B008
        None, "--env", "-e", help="Environment variables (KEY=VALUE)"
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
    session: str = typer.Option(
        None, "--session", "-s", help="Session ID for persistent execution"
    ),
) -> None:
    """Execute Python code in the safe sandbox environment.

    You can provide code directly as an argument or from a file via --file.
    Valid tools can be imported directly in the code.

    Use --session <name> to maintain state between runs.

    Example:
        locca sandbox run "print('Hello World')"
        locca sandbox run --session my-session "x=1"
        locca sandbox run --session my-session "print(x)"
    """
    code = code_arg or code_opt
    if not code and not file:
        typer.echo("Error: Must provide either CODE argument or --file", err=True)
        raise typer.Exit(code=1)
    if code and file:
        typer.echo("Error: Cannot provide both CODE and --file", err=True)
        raise typer.Exit(code=1)
    if file:
        file_path = Path(file)
        code = file_path.read_text(encoding="utf-8")
    env_vars = _parse_env(env)
    level = getattr(logging, log_level.upper(), logging.INFO)
    ctx = bootstrap(log_level=level)
    sandbox_manager = ctx.get("sandbox")
    if not sandbox_manager:
        typer.echo("Error: Sandbox manager not available", err=True)
        raise typer.Exit(code=1)
    persistence = True
    session_id = session or "default"
    asyncio.run(
        _run_code(
            sandbox_manager,
            code,
            session_id=session_id,
            env_vars=env_vars,
            timeout=timeout,
            persistence=persistence,
        )
    )


@app.command(name="exec")
@safe_entrypoint("cli.sandbox.exec")
def exec_cmd(
    command: str = typer.Argument(..., help="Shell command to execute"),
    timeout: int = typer.Option(30, help="Execution timeout in seconds"),
    env: list[str] = typer.Option(  # noqa: B008
        None, "--env", "-e", help="Environment variables (KEY=VALUE)"
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
    session: str = typer.Option(
        None, "--session", "-s", help="Session ID for persistent execution"
    ),
) -> None:
    """Execute a shell command in the sandbox.

    This wraps the shell command in a Python subprocess call.
    """
    python_code = f"""import subprocess
import sys

cmd = {command!r}
try:
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout={timeout})
    print(result.stdout, end='')
    print(result.stderr, file=sys.stderr, end='')
    if result.returncode != 0:
        pass
except Exception as e:
    print(f"Error executing shell command: {{e}}", file=sys.stderr)"""
    env_vars = _parse_env(env)
    level = getattr(logging, log_level.upper(), logging.INFO)
    ctx = bootstrap(log_level=level)
    sandbox_manager = ctx.get("sandbox")
    if not sandbox_manager:
        typer.echo("Error: Sandbox manager not available", err=True)
        raise typer.Exit(code=1)
    persistence = True
    session_id = session or "default"
    asyncio.run(
        _run_code(
            sandbox_manager,
            python_code,
            session_id=session_id,
            env_vars=env_vars,
            timeout=timeout,
            persistence=persistence,
        )
    )


@app.command(name="stop")
@safe_entrypoint("cli.sandbox.stop")
def stop_session(
    session_id: str = typer.Argument(..., help="Session ID to stop"),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
) -> None:
    """Stop a persistent sandbox session."""

    async def _stop(sandbox_manager: Any, sid: str):
        sandbox = sandbox_manager.get_sandbox()
        await sandbox.stop_session(sid)

    level = getattr(logging, log_level.upper(), logging.INFO)
    ctx = bootstrap(log_level=level)
    sandbox_manager = ctx.get("sandbox")
    if not sandbox_manager:
        typer.echo("Error: Sandbox manager not available", err=True)
        raise typer.Exit(code=1)
    asyncio.run(_stop(sandbox_manager, session_id))
    typer.echo(f"Session '{session_id}' stopped.")
