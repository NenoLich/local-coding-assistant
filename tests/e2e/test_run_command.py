"""End-to-end tests for the run command functionality."""

from unittest.mock import patch

from local_coding_assistant.cli.main import app


class TestRunCommand:
    """Test cases for the run query command."""

    def test_run_query_basic(self, cli_runner, mock_env_vars, mock_bootstrap_success):
        """Test basic run query functionality."""
        mock_bootstrap, mock_ctx = mock_bootstrap_success

        result = cli_runner.invoke(app, ["run", "query", "Hello world"])

        assert result.exit_code == 0
        assert "Running query: Hello world" in result.stdout
        assert "Response:" in result.stdout
        assert "[LLMManager] Echo: test query" in result.stdout

        # Verify bootstrap was called
        mock_bootstrap.assert_called_once()

    def test_run_query_verbose(self, cli_runner, mock_env_vars, mock_bootstrap_success):
        """Test run query with verbose flag."""
        mock_bootstrap, mock_ctx = mock_bootstrap_success

        result = cli_runner.invoke(app, ["run", "query", "Hello world", "--verbose"])

        assert result.exit_code == 0
        assert "Running query: Hello world" in result.stdout
        assert "Verbose mode enabled" in result.stdout
        assert "Response:" in result.stdout
        assert "[LLMManager] Echo: test query" in result.stdout

    def test_run_query_with_model(
        self, cli_runner, mock_env_vars, mock_bootstrap_success
    ):
        """Test run query with specific model."""
        mock_bootstrap, mock_ctx = mock_bootstrap_success

        result = cli_runner.invoke(
            app, ["run", "query", "Hello world", "--model", "gpt-4"]
        )

        assert result.exit_code == 0
        assert "Running query: Hello world" in result.stdout
        assert "Using model: gpt-4" in result.stdout
        assert "Response:" in result.stdout
        assert "[LLMManager] Echo: test query" in result.stdout

    def test_run_query_with_log_level(
        self, cli_runner, mock_env_vars, mock_bootstrap_success
    ):
        """Test run query with custom log level."""
        mock_bootstrap, mock_ctx = mock_bootstrap_success

        result = cli_runner.invoke(
            app, ["run", "query", "Hello world", "--log-level", "DEBUG"]
        )

        assert result.exit_code == 0
        assert "Running query: Hello world" in result.stdout
        assert "Response:" in result.stdout
        assert "[LLMManager] Echo: test query" in result.stdout

    def test_run_query_combined_options(
        self, cli_runner, mock_env_vars, mock_bootstrap_success
    ):
        """Test run query with all options combined."""
        mock_bootstrap, mock_ctx = mock_bootstrap_success

        result = cli_runner.invoke(
            app,
            [
                "run",
                "query",
                "Hello world",
                "--verbose",
                "--model",
                "gpt-4",
                "--log-level",
                "DEBUG",
            ],
        )

        assert result.exit_code == 0
        assert "Running query: Hello world" in result.stdout
        assert "Verbose mode enabled" in result.stdout
        assert "Using model: gpt-4" in result.stdout
        assert "Response:" in result.stdout
        assert "[LLMManager] Echo: test query" in result.stdout

    def test_run_query_error_handling(self, cli_runner, mock_env_vars):
        """Test run query error handling."""
        # Mock bootstrap failure
        with patch(
            "local_coding_assistant.cli.commands.run.bootstrap"
        ) as mock_bootstrap:
            mock_bootstrap.return_value = {"runtime": None}

            result = cli_runner.invoke(app, ["run", "query", "Hello world"])

            assert result.exit_code == 1
            assert "Error: Runtime manager not available" in result.stdout

    def test_run_query_long_text(
        self, cli_runner, mock_env_vars, mock_bootstrap_success
    ):
        """Test run query with long text input."""
        mock_bootstrap, mock_ctx = mock_bootstrap_success

        long_query = "This is a very long query that should still work properly with the CLI system and be processed correctly by the runtime manager"

        result = cli_runner.invoke(app, ["run", "query", long_query])

        assert result.exit_code == 0
        assert f"Running query: {long_query}" in result.stdout
        assert "Response:" in result.stdout
        assert "[LLMManager] Echo: test query" in result.stdout

    def test_run_query_special_characters(
        self, cli_runner, mock_env_vars, mock_bootstrap_success
    ):
        """Test run query with special characters."""
        mock_bootstrap, mock_ctx = mock_bootstrap_success

        special_query = (
            "Query with \"quotes\" and 'apostrophes' and special chars: @#$%^&*()"
        )

        result = cli_runner.invoke(app, ["run", "query", special_query])

        assert result.exit_code == 0
        assert f"Running query: {special_query}" in result.stdout
        assert "Response:" in result.stdout
        assert "[LLMManager] Echo: test query" in result.stdout
