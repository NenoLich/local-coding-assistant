"""Configure system settings."""

import os

import typer

from local_coding_assistant.core.error_handler import safe_entrypoint
from local_coding_assistant.utils.logging import get_logger

app = typer.Typer(name="config", help="Configure system settings")
log = get_logger("cli.config")

_PREFIX = "LOCCA_"


def _k(key: str) -> str:
    return key if key.startswith(_PREFIX) else f"{_PREFIX}{key}"


@app.command("get")
@safe_entrypoint("cli.config.get")
def get_config(
    key: str | None = typer.Argument(None, help="Configuration key to get"),
) -> None:
    """Get configuration value(s) from environment (prefix LOCCA_)."""
    # Logging level can be honored by bootstrap if needed; here we just use module logger
    if key:
        env_key = _k(key)
        val = os.environ.get(env_key)
        if val is None:
            typer.echo(f"{env_key} is not set")
        else:
            typer.echo(f"{env_key}={val}")
    else:
        typer.echo("All configuration (env, LOCCA_*):")
        for k, v in sorted(os.environ.items()):
            if k.startswith(_PREFIX):
                typer.echo(f"{k}={v}")


@app.command("set")
@safe_entrypoint("cli.config.set")
def set_config(
    key: str = typer.Argument(..., help="Configuration key to set"),
    value: str = typer.Argument(..., help="Value to set"),
) -> None:
    """Set a configuration value in the current process environment (prefix LOCCA_)."""
    env_key = _k(key)
    os.environ[env_key] = value
    log.info("Set %s", env_key)
    typer.echo(f"Set {env_key}={value}")
