"""End-to-end tests for config CLI commands."""

import os
from unittest.mock import patch

from local_coding_assistant.cli.main import app
from local_coding_assistant.config.env_manager import get_env_manager


class TestConfigCommands:
    """Test cases for configuration CLI commands."""

    def test_config_get_nonexistent_key(self, cli_runner):
        """Test getting a configuration key that doesn't exist."""
        result = cli_runner.invoke(app, ["config", "get", "NONEXISTENT_KEY"])

        assert result.exit_code == 1
        assert "No configuration found for key: LOCCA_NONEXISTENT_KEY" in result.stderr

    def test_config_get_existing_key(self, cli_runner, mock_env_vars):
        """Test getting a configuration key that exists."""
        # Set a test environment variable
        os.environ["LOCCA_CUSTOM_KEY"] = "custom_value"

        result = cli_runner.invoke(app, ["config", "get", "CUSTOM_KEY"])

        assert result.exit_code == 0
        assert "LOCCA_CUSTOM_KEY=custom_value" in result.stdout

    @patch("local_coding_assistant.cli.commands.config.get_env_manager")
    def test_config_get_all(self, mock_get_env_manager, cli_runner, mock_env_vars):
        """Test getting all configuration values."""
        # Ensure our mock environment variables are set
        assert os.environ.get("LOCCA_TEST_MODE") == "true"
        assert os.environ.get("LOCCA_LOG_LEVEL") == "INFO"

        # Add a test-specific variable
        test_key = f"LOCCA_TEST_{os.getpid()}"
        test_value = "test_value"
        os.environ[test_key] = test_value

        result = cli_runner.invoke(app, ["config", "get"])

        # Check for the expected output
        assert result.exit_code == 0
        assert "All configuration (LOCCA_*):" in result.stdout

        # Verify our test variables are shown correctly
        assert test_key in result.stdout
        assert "LOCCA_TEST_MODE" in result.stdout
        assert "LOCCA_LOG_LEVEL" in result.stdout

    @patch("local_coding_assistant.cli.commands.config.get_env_manager")
    def test_config_set_new_key(self, mock_get_env_manager, cli_runner, tmp_path):
        """Test setting a new configuration key."""
        result = cli_runner.invoke(app, ["config", "set", "NEW_KEY", "new_value"])

        assert result.exit_code == 0

        # The implementation now logs the set operation
        assert "Set and persisted LOCCA_NEW_KEY" in result.stdout

        # Verify the environment variable was set in the current process
        assert os.environ.get("LOCCA_NEW_KEY") == "new_value"

    @patch("local_coding_assistant.cli.commands.config.get_env_manager")
    def test_config_set_overwrite_existing(self, mock_get_env_manager, cli_runner, tmp_path):
        """Test setting a configuration key that already exists."""
        # Pre-set a value in the environment
        os.environ["LOCCA_EXISTING_KEY"] = "old_value"

        result = cli_runner.invoke(app, ["config", "set", "EXISTING_KEY", "new_value"])

        assert result.exit_code == 0

        # The implementation now logs the set operation
        assert "Set and persisted LOCCA_EXISTING_KEY" in result.stdout

        # Verify the environment variable was updated
        assert os.environ.get("LOCCA_EXISTING_KEY") == "new_value"

    @patch("local_coding_assistant.cli.commands.config.get_env_manager")
    def test_config_set_special_characters(self, mock_get_env_manager, cli_runner, tmp_path):
        """Test setting a configuration key with special characters in value."""
        special_value = 'Value with "quotes" and spaces and special chars: @#$%^&*()'

        result = cli_runner.invoke(app, ["config", "set", "SPECIAL_KEY", special_value])

        assert result.exit_code == 0

        # The implementation now logs the set operation
        assert "Set and persisted LOCCA_SPECIAL_KEY" in result.stdout

        # Verify the environment variable was set correctly
        assert os.environ.get("LOCCA_SPECIAL_KEY") == special_value

    @patch("local_coding_assistant.cli.commands.config.get_env_manager")
    def test_config_set_empty_value(self, mock_get_env_manager, cli_runner, tmp_path):
        """Test setting a configuration key with empty value."""
        result = cli_runner.invoke(app, ["config", "set", "EMPTY_KEY", ""])

        assert result.exit_code == 0

        # The implementation now logs the set operation
        assert "Set and persisted LOCCA_EMPTY_KEY" in result.stdout

        # Verify the environment variable was set to empty string
        assert os.environ.get("LOCCA_EMPTY_KEY") == ""

    @patch("local_coding_assistant.cli.commands.config.get_env_manager")
    def test_config_workflow(self, mock_get_env_manager, cli_runner, tmp_path):
        """Test a complete config workflow: set, get, verify."""
        # Set a value
        result = cli_runner.invoke(
            app, ["config", "set", "WORKFLOW_TEST", "test_value"]
        )
        assert result.exit_code == 0

        # Get the value - it might show as from environment or .env.local
        result = cli_runner.invoke(app, ["config", "get", "WORKFLOW_TEST"])
        assert result.exit_code == 0
        assert "LOCCA_WORKFLOW_TEST=test_value" in result.stdout
        assert any(
            source in result.stdout
            for source in ["(from .env.local)", "(from environment)"]
        )

        # Get all values and verify it's included
        result = cli_runner.invoke(app, ["config", "get"])
        assert result.exit_code == 0
        assert "LOCCA_WORKFLOW_TEST=test_value" in result.stdout
        # Check that the value appears in the output with either source
        assert any(
            "LOCCA_WORKFLOW_TEST=test_value (from " in line
            for line in result.stdout.splitlines()
            if line.startswith("LOCCA_WORKFLOW_TEST")
        )
