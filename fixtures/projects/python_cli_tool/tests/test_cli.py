"""Tests for the Python CLI tool."""

import json
import tempfile
from pathlib import Path

from click.testing import CliRunner
from python_cli_tool import Config, cli


class TestConfig:
    """Tests for configuration management."""

    def test_default_config(self):
        """Test default configuration values."""
        config = Config()
        assert config.get("api_url") == "https://api.example.com"
        assert config.get("timeout") == 30
        assert config.get("retries") == 3
        assert config.get("output_format") == "json"

    def test_config_override(self):
        """Test configuration override."""
        config = Config()
        config.set("timeout", 60)
        assert config.get("timeout") == 60

    def test_config_save_load(self):
        """Test saving and loading configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config = Config(str(config_path))
            config.set("api_url", "https://custom.api.com")
            config.save()

            # Load fresh config
            config2 = Config(str(config_path))
            assert config2.get("api_url") == "https://custom.api.com"

    def test_env_override(self, monkeypatch):
        """Test environment variable override."""
        monkeypatch.setenv("PYTHON_CLI_TOOL_TIMEOUT", "120")
        monkeypatch.setenv("PYTHON_CLI_TOOL_RETRIES", "5")
        config = Config()
        assert config.get("timeout") == 120
        assert config.get("retries") == 5


class TestCLI:
    """Tests for CLI commands."""

    def setup_method(self):
        self.runner = CliRunner()

    def test_version(self):
        """Test version command."""
        result = self.runner.invoke(cli, ["version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_greet(self):
        """Test greet command."""
        result = self.runner.invoke(cli, ["greet", "--name", "Alice"])
        assert result.exit_code == 0
        assert "Hello, Alice!" in result.output

    def test_greet_default(self):
        """Test greet command with default name."""
        result = self.runner.invoke(cli, ["greet"])
        assert result.exit_code == 0
        assert "Hello, World!" in result.output

    def test_config_show(self):
        """Test config show command."""
        result = self.runner.invoke(cli, ["config-show"])
        assert result.exit_code == 0
        config = json.loads(result.output)
        assert config["api_url"] == "https://api.example.com"

    def test_config_set(self):
        """Test config set command."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(cli, ["config-set", "timeout", "45"])
            assert result.exit_code == 0
            assert "Set timeout = 45" in result.output

            # Verify it persisted
            result2 = self.runner.invoke(cli, ["config-show"])
            config = json.loads(result2.output)
            assert config["timeout"] == 45

    def test_process_file(self):
        """Test process command."""
        with self.runner.isolated_filesystem():
            input_file = Path("input.txt")
            input_file.write_text("line 1\nline 2\nline 3\n")

            result = self.runner.invoke(cli, ["process", str(input_file)])
            assert result.exit_code == 0
            output = json.loads(result.output)
            assert output["lines"] == 4  # 3 newlines + 1
            assert output["size"] > 0

    def test_process_file_with_output(self):
        """Test process command with output file."""
        with self.runner.isolated_filesystem():
            input_file = Path("input.txt")
            input_file.write_text("test content")
            output_file = Path("output.json")

            result = self.runner.invoke(cli, ["process", str(input_file), "-o", str(output_file)])
            assert result.exit_code == 0
            assert output_file.exists()

            output_data = json.loads(output_file.read_text())
            assert output_data["input_file"] == "input.txt"

    def test_fetch(self):
        """Test fetch command."""
        result = self.runner.invoke(cli, ["fetch", "https://example.com", "https://test.com"])
        assert result.exit_code == 0
        assert "https://example.com" in result.output
        assert "https://test.com" in result.output
        assert "200 OK" in result.output


class TestIntegration:
    """Integration tests."""

    def setup_method(self):
        self.runner = CliRunner()

    def test_full_workflow(self):
        """Test a full workflow: config -> process -> fetch."""
        with self.runner.isolated_filesystem():
            # Set config
            result = self.runner.invoke(cli, ["config-set", "output_format", "yaml"])
            assert result.exit_code == 0

            # Create input file
            input_file = Path("data.txt")
            input_file.write_text("hello\nworld\n")

            # Process it
            result = self.runner.invoke(cli, ["process", str(input_file)])
            assert result.exit_code == 0
            output = json.loads(result.output)
            assert output["config_used"] == "yaml"
            assert output["lines"] == 3  # 2 newlines + 1

            # Fetch some URLs
            result = self.runner.invoke(cli, ["fetch", "https://api.example.com/users"])
            assert result.exit_code == 0