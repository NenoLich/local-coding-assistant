import json
from typer.testing import CliRunner

from local_coding_assistant.cli.main import app


def test_cli_list_tools_text_and_json():
    runner = CliRunner()

    # Text output
    res = runner.invoke(
        app,
        [
            "list-tools",
            "list",
            "--log-level",
            "DEBUG",
        ],
    )
    assert res.exit_code == 0
    assert "Available tools:" in res.stdout

    # JSON output
    res_json = runner.invoke(
        app,
        [
            "list-tools",
            "list",
            "--json",
            "--log-level",
            "INFO",
        ],
    )
    assert res_json.exit_code == 0
    data = json.loads(res_json.stdout.strip())
    assert "count" in data and "tools" in data
    assert isinstance(data["tools"], list)


def test_cli_config_set_and_get_roundtrip():
    runner = CliRunner()
    env = {}

    # Set
    res_set = runner.invoke(
        app,
        [
            "config",
            "set",
            "API_KEY",
            "secret",
            "--log-level",
            "WARNING",
        ],
        env=env,
    )
    assert res_set.exit_code == 0
    assert "Set LOCCA_API_KEY=secret" in res_set.stdout

    # Get single
    res_get = runner.invoke(
        app,
        [
            "config",
            "get",
            "API_KEY",
        ],
        env=env,
    )
    assert res_get.exit_code == 0
    assert "LOCCA_API_KEY=secret" in res_get.stdout


def test_cli_serve_start_outputs():
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "serve",
            "start",
            "--host",
            "0.0.0.0",
            "--port",
            "8080",
            "--reload",
            "--log-level",
            "ERROR",
        ],
    )
    assert res.exit_code == 0
    assert "Starting server on 0.0.0.0:8080" in res.stdout
    assert "Auto-reload enabled" in res.stdout


def test_cli_validation_errors_missing_and_type():
    runner = CliRunner()

    # Missing required argument for run query -> exit code 2 and usage error
    res_missing = runner.invoke(
        app,
        [
            "run",
            "query",
            # missing TEXT argument
        ],
    )
    assert res_missing.exit_code == 2
    assert (
        "Missing argument 'TEXT'" in res_missing.stdout
        or "Missing argument 'TEXT'" in res_missing.stderr
    )

    # Invalid type for --port (expects int) -> exit code 2
    res_type = runner.invoke(
        app,
        [
            "serve",
            "start",
            "--port",
            "not-a-number",
        ],
    )
    assert res_type.exit_code == 2
    # Click/Typer error message may vary by version; check a generic phrase
    assert (
        "Invalid value for '--port'" in res_type.stdout
        or "Invalid value for '--port'" in res_type.stderr
    )
