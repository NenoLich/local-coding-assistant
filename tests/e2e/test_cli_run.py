from typer.testing import CliRunner

from local_coding_assistant.cli.main import app


def test_cli_run_query_end_to_end():
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "run",
            "query",
            "Hello world",
            "--verbose",
            "--model",
            "dummy-model",
        ],
    )

    assert result.exit_code == 0

    # From run.query
    assert "Running query: Hello world" in result.stdout
    assert "Verbose mode enabled" in result.stdout
    assert "Using model: dummy-model" in result.stdout

    # Response header and echo from LLMManager
    assert "Response:" in result.stdout
    assert "[LLMManager] Echo: Hello world" in result.stdout
