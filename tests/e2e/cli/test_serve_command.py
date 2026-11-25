"""End-to-end tests for serve CLI command."""

from local_coding_assistant.cli.main import app


class TestServeCommand:
    """Test cases for the serve CLI command."""

    def test_serve_start_basic(self, cli_runner, mock_bootstrap_serve):
        """Test basic serve start functionality."""
        mock_bootstrap, _mock_ctx = mock_bootstrap_serve

        result = cli_runner.invoke(app, ["serve", "start"])

        assert result.exit_code == 0
        assert "Server starting on 127.0.0.1:8001" in result.stdout

        # Verify bootstrap was called
        mock_bootstrap.assert_called_once()

    def test_serve_start_custom_host_port(self, cli_runner, mock_bootstrap_serve):
        """Test serve start with custom host and port."""
        _mock_bootstrap, _mock_ctx = mock_bootstrap_serve

        result = cli_runner.invoke(
            app, ["serve", "start", "--host", "127.0.0.1", "--port", "9000"]
        )

        assert result.exit_code == 0
        assert "Server starting on 127.0.0.1:9000" in result.stdout

    def test_serve_start_with_log_level(self, cli_runner, mock_bootstrap_serve):
        """Test serve start with custom log level."""
        _mock_bootstrap, _mock_ctx = mock_bootstrap_serve

        result = cli_runner.invoke(app, ["serve", "start", "--log-level", "DEBUG"])

        assert result.exit_code == 0
        assert "Server starting on 127.0.0.1:8001" in result.stdout

    def test_serve_start_combined_options(self, cli_runner, mock_bootstrap_serve):
        """Test serve start with all options combined."""
        _mock_bootstrap, _mock_ctx = mock_bootstrap_serve

        result = cli_runner.invoke(
            app,
            [
                "serve",
                "start",
                "--host",
                "127.0.0.1",
                "--port",
                "9000",
                "--log-level",
                "DEBUG",
            ],
        )

        assert result.exit_code == 0
        assert "Server starting on 127.0.0.1:9000" in result.stdout

    def test_serve_start_runtime_info(self, cli_runner, mock_bootstrap_serve):
        """Test that serve start shows runtime information."""
        _mock_bootstrap, _mock_ctx = mock_bootstrap_serve

        result = cli_runner.invoke(app, ["serve", "start"])

        assert result.exit_code == 0
        # The serve command should show runtime component info
        # Note: This is a placeholder test since the actual server implementation is not complete

    def test_serve_start_ipv6_host(self, cli_runner, mock_bootstrap_serve):
        """Test serve start with IPv6 host."""
        _mock_bootstrap, _mock_ctx = mock_bootstrap_serve

        result = cli_runner.invoke(
            app, ["serve", "start", "--host", "::1", "--port", "8080"]
        )

        assert result.exit_code == 0
        assert "Server starting on ::1:8080" in result.stdout
