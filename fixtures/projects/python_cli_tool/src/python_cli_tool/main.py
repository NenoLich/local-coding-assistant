"""Command implementations."""

import json
from pathlib import Path

import click

from python_cli_tool import Config, load_config


@click.group()
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to config file")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.pass_context
def cli(ctx: click.Context, config: str | None, verbose: bool) -> None:
    """Python CLI Tool - A simple CLI for testing."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)
    ctx.obj["verbose"] = verbose


@cli.command()
@click.pass_context
def version(ctx: click.Context) -> None:
    """Show version information."""
    from python_cli_tool import __version__

    click.echo(f"python-cli-tool version {__version__}")


@cli.command()
@click.pass_context
def config_show(ctx: click.Context) -> None:
    """Show current configuration."""
    config: Config = ctx.obj["config"]
    click.echo(json.dumps(config.all, indent=2))


@cli.command()
@click.argument("key")
@click.argument("value")
@click.pass_context
def config_set(ctx: click.Context, key: str, value: str) -> None:
    """Set a configuration value."""
    config: Config = ctx.obj["config"]
    parsed_value = config._parse_value(value)
    config.set(key, parsed_value)
    config.save()
    click.echo(f"Set {key} = {parsed_value}")


@cli.command()
@click.option("--name", "-n", default="World", help="Name to greet")
@click.pass_context
def greet(ctx: click.Context, name: str) -> None:
    """Greet someone."""
    click.echo(f"Hello, {name}!")


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.pass_context
def process(ctx: click.Context, input_file: str, output: str | None) -> None:
    """Process a file (placeholder command)."""
    config: Config = ctx.obj["config"]
    verbose = ctx.obj["verbose"]

    input_path = Path(input_file)
    content = input_path.read_text()

    if verbose:
        click.echo(f"Processing {input_path} (size: {len(content)} bytes)")

    # Simulate processing
    result = {
        "input_file": str(input_path),
        "size": len(content),
        "lines": content.count("\n") + 1,
        "config_used": config.get("output_format"),
    }

    output_data = json.dumps(result, indent=2)

    if output:
        Path(output).write_text(output_data)
        click.echo(f"Output written to {output}")
    else:
        click.echo(output_data)


@cli.command()
@click.argument("urls", nargs=-1, required=True)
@click.pass_context
def fetch(ctx: click.Context, urls: tuple[str, ...]) -> None:
    """Fetch URLs (placeholder - simulates API calls)."""
    config: Config = ctx.obj["config"]
    api_url = config.get("api_url")
    timeout = config.get("timeout")

    click.echo(f"Using API: {api_url}")
    click.echo(f"Timeout: {timeout}s")

    for url in urls:
        click.echo(f"Fetching {url}... (simulated)")
        # In real implementation, would use httpx or requests
        click.echo("  -> 200 OK (mock response)")


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()