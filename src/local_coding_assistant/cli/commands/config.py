"""Configure system settings with file-based persistence."""

import os

import typer

from local_coding_assistant.config.env_manager import EnvManager
from local_coding_assistant.core.error_handler import safe_entrypoint
from local_coding_assistant.utils.logging import get_logger

# Initialize EnvManager with the default prefix
_env_manager = EnvManager()

app = typer.Typer(name="config", help="Configure system settings")
log = get_logger("cli.config")


def _k(key: str) -> str:
    """Compatibility wrapper for backward compatibility.

    Note: Prefer using _env_manager.set_env() and _env_manager.get_env() directly,
    which handle prefixing automatically.
    """
    return _env_manager.with_prefix(key)


def _load_env() -> dict[str, str]:
    """Load all environment variables with the configured prefix.

    Returns:
        Dictionary of environment variables with the configured prefix
    """
    # Load .env files first
    _env_manager.load_env_files()

    # Get all env vars with the prefix
    return _env_manager.get_config_from_env()


@app.command("get")
@safe_entrypoint("cli.config.get")
def get_config(
    key: str | None = typer.Argument(None, help="Configuration key to get"),
) -> None:
    """
    Get configuration value(s) from environment (prefix LOCCA_).

    Checks in this order:
    1. System environment variables
    2. .env.local file
    """
    if key:
        # Get value using EnvManager which handles prefixing
        env_key = _k(key)
        val = _env_manager.get_env(key)
        if val is None:
            typer.echo(f"No configuration found for key: {env_key}", err=True)
            raise typer.Exit(1)
        else:
            # Check source by comparing with raw environment
            source = "environment" if env_key in os.environ else ".env.local"
            typer.echo(f"{env_key}={val} (from {source})")
    else:
        # Get all configuration from EnvManager
        prefix = _env_manager.env_prefix
        typer.echo(f"All configuration ({prefix}*):")

        # Get config from environment (this includes both .env and os.environ)
        _env_manager.load_env_files()  # Ensure .env files are loaded

        # Display all found variables with their sources
        for k, v in sorted(_env_manager.get_all_env_vars().items()):
            if k.startswith(prefix):
                source = "environment" if k in os.environ else ".env.local"
                typer.echo(f"{k}={v} (from {source})")


@app.command("set")
@safe_entrypoint("cli.config.set")
def set_config(
    key: str = typer.Argument(..., help="Configuration key to set"),
    value: str = typer.Argument(..., help="Value to set"),
    temporary: bool = typer.Option(
        False,
        "--temporary",
        "-t",
        help="Set only in current process, don't persist to .env.local",
    ),
) -> None:
    """
    Set a configuration value.

    By default, saves to .env.local for persistence between sessions.
    Use --temporary to set only in the current process.
    """
    if not temporary:
        # Set in environment and persist to file
        _env_manager.set_env(key, value)
        _env_manager.save_to_env_file(_env_manager.with_prefix(key), value)
        log.info("Set and persisted %s", _env_manager.with_prefix(key))
    else:
        # Set only in current process
        _env_manager.set_env(key, value)
        log.info("Temporarily set %s (not persisted)", _k(key))


@app.command("unset")
@safe_entrypoint("cli.config.unset")
def unset_config(
    key: str = typer.Argument(..., help="Configuration key to unset"),
    temporary: bool = typer.Option(
        False,
        "--temporary",
        "-t",
        help="Unset only in current process, don't modify .env.local",
    ),
) -> None:
    """
    Unset a configuration value.

    By default, removes from both environment and .env.local.
    Use --temporary to unset only in the current process.
    """
    if not temporary:
        # Remove from both environment and file
        _env_manager.unset_env(key)
        _env_manager.remove_from_env_file(_env_manager.with_prefix(key))
        prefixed_key = _env_manager.with_prefix(key)
        log.info("Unset and removed %s from persistence", prefixed_key)
        typer.echo(f"Unset and removed {prefixed_key} from persistence")
    else:
        # Remove only from current process
        _env_manager.unset_env(key)
        prefixed_key = _env_manager.with_prefix(key)
        log.info("Temporarily unset %s (still in .env.local)", prefixed_key)
        typer.echo(f"Temporarily unset {prefixed_key} (still in .env.local)")
