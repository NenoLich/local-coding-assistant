"""End-to-end tests for serve CLI command."""


from local_coding_assistant.cli.main import app


class TestServeCommand:
    """Test cases for the serve CLI command."""

    def test_serve_start_basic(self, cli_runner, mock_bootstrap_serve):
        """Test basic serve start functionality."""
        mock_bootstrap, mock_ctx = mock_bootstrap_serve

        result = cli_runner.invoke(app, ["serve", "start"])

        assert result.exit_code == 0
        assert "Starting server on 127.0.0.1:8000" in result.stdout

        # Verify bootstrap was called
        mock_bootstrap.assert_called_once()

    def test_serve_start_custom_host_port(self, cli_runner, mock_bootstrap_serve):
        """Test serve start with custom host and port."""
        mock_bootstrap, mock_ctx = mock_bootstrap_serve

        result = cli_runner.invoke(app, [
            "serve",
            "start",
            "--host",
            "0.0.0.0",
            "--port",
            "9000"
        ])

        assert result.exit_code == 0
        assert "Starting server on 0.0.0.0:9000" in result.stdout

    def test_serve_start_with_reload(self, cli_runner, mock_bootstrap_serve):
        """Test serve start with auto-reload enabled."""
        mock_bootstrap, mock_ctx = mock_bootstrap_serve

        result = cli_runner.invoke(app, [
            "serve",
            "start",
            "--reload"
        ])

        assert result.exit_code == 0
        assert "Starting server on 127.0.0.1:8000" in result.stdout
        assert "Auto-reload enabled" in result.stdout

    def test_serve_start_with_log_level(self, cli_runner, mock_bootstrap_serve):
        """Test serve start with custom log level."""
        mock_bootstrap, mock_ctx = mock_bootstrap_serve

        result = cli_runner.invoke(app, [
            "serve",
            "start",
            "--log-level",
            "DEBUG"
        ])

        assert result.exit_code == 0
        assert "Starting server on 127.0.0.1:8000" in result.stdout

    def test_serve_start_combined_options(self, cli_runner, mock_bootstrap_serve):
        """Test serve start with all options combined."""
        mock_bootstrap, mock_ctx = mock_bootstrap_serve

        result = cli_runner.invoke(app, [
            "serve",
            "start",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--reload",
            "--log-level",
            "DEBUG"
        ])

        assert result.exit_code == 0
        assert "Starting server on 0.0.0.0:9000" in result.stdout
        assert "Auto-reload enabled" in result.stdout

    def test_serve_start_runtime_info(self, cli_runner, mock_bootstrap_serve):
        """Test that serve start shows runtime information."""
        mock_bootstrap, mock_ctx = mock_bootstrap_serve

        result = cli_runner.invoke(app, ["serve", "start"])

        assert result.exit_code == 0
        # The serve command should show runtime component info
        # Note: This is a placeholder test since the actual server implementation is not complete

    def test_serve_start_ipv6_host(self, cli_runner, mock_bootstrap_serve):
        """Test serve start with IPv6 host."""
        mock_bootstrap, mock_ctx = mock_bootstrap_serve

        result = cli_runner.invoke(app, [
            "serve",
            "start",
            "--host",
            "::1",
            "--port",
            "8080"
        ])

        assert result.exit_code == 0
        assert "Starting server on ::1:8080" in result.stdout
