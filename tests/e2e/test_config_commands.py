"""End-to-end tests for config CLI commands."""

import os
from unittest.mock import patch

from local_coding_assistant.cli.main import app
from local_coding_assistant.config.env_manager import EnvManager


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

    def test_config_get_all(self, cli_runner, mock_env_vars):
        """Test getting all configuration values."""
        # Set additional test environment variables
        os.environ["LOCCA_ANOTHER_KEY"] = "another_value"

        result = cli_runner.invoke(app, ["config", "get"])

        assert result.exit_code == 0
        assert "All configuration (LOCCA_*):" in result.stdout
        assert "LOCCA_TEST_MODE=true (from environment)" in result.stdout
        assert "LOCCA_LOG_LEVEL=INFO (from environment)" in result.stdout
        assert "LOCCA_ANOTHER_KEY=another_value (from environment)" in result.stdout

    @patch.object(EnvManager, "_get_default_env_paths")
    def test_config_set_new_key(self, mock_default_paths, cli_runner, tmp_path):
        """Test setting a new configuration key."""
        # Create a temporary .env.local file
        env_file = tmp_path / ".env.local"

        # Mock the default paths to use our temporary directory
        mock_default_paths.return_value = [env_file]

        # We need to re-initialize the EnvManager to use our mocked paths
        from local_coding_assistant.cli.commands import config

        config._env_manager = EnvManager(env_paths=[env_file])

        result = cli_runner.invoke(app, ["config", "set", "NEW_KEY", "new_value"])

        assert result.exit_code == 0

        # The actual implementation doesn't output anything on success for set command
        assert result.stdout.strip() == ""

        # Verify the environment variable was set in the current process
        assert os.environ.get("LOCCA_NEW_KEY") == "new_value"

        # Verify the value was written to the .env.local file
        assert env_file.exists()
        with open(env_file) as f:
            content = f.read()
            assert "LOCCA_NEW_KEY=new_value" in content

    @patch.object(EnvManager, "_get_default_env_paths")
    def test_config_set_overwrite_existing(
        self, mock_default_paths, cli_runner, tmp_path
    ):
        """Test setting a configuration key that already exists."""
        # Create a temporary .env.local file
        env_file = tmp_path / ".env.local"

        # Mock the default paths to use our temporary directory
        mock_default_paths.return_value = [env_file]

        # We need to re-initialize the EnvManager to use our mocked paths
        from local_coding_assistant.cli.commands import config

        config._env_manager = EnvManager(env_paths=[env_file])

        # Pre-set a value in the environment and in the .env.local file
        os.environ["LOCCA_EXISTING_KEY"] = "old_value"
        with open(env_file, "w") as f:
            f.write("LOCCA_EXISTING_KEY=old_value\n")

        result = cli_runner.invoke(app, ["config", "set", "EXISTING_KEY", "new_value"])

        assert result.exit_code == 0

        # The actual implementation doesn't output anything on success for set command
        assert result.stdout.strip() == ""

        # Verify the environment variable was updated
        assert os.environ.get("LOCCA_EXISTING_KEY") == "new_value"

        # Verify the value was updated in the .env.local file
        with open(env_file) as f:
            content = f.read()
            assert "LOCCA_EXISTING_KEY=new_value" in content

    @patch.object(EnvManager, "_get_default_env_paths")
    def test_config_set_special_characters(
        self, mock_default_paths, cli_runner, tmp_path
    ):
        """Test setting a configuration key with special characters in value."""
        # Create a temporary .env.local file
        env_file = tmp_path / ".env.local"

        # Mock the default paths to use our temporary directory
        mock_default_paths.return_value = [env_file]

        # We need to re-initialize the EnvManager to use our mocked paths
        from local_coding_assistant.cli.commands import config

        config._env_manager = EnvManager(env_paths=[env_file])

        special_value = 'Value with "quotes" and spaces and special chars: @#$%^&*()'

        result = cli_runner.invoke(app, ["config", "set", "SPECIAL_KEY", special_value])

        assert result.exit_code == 0

        # The actual implementation doesn't output anything on success for set command
        assert result.stdout.strip() == ""

        # Verify the environment variable was set correctly
        assert os.environ.get("LOCCA_SPECIAL_KEY") == special_value

        # Verify the value was written to the .env.local file with proper escaping
        assert env_file.exists()
        with open(env_file) as f:
            content = f.read()
            assert f"LOCCA_SPECIAL_KEY={special_value}" in content

    @patch.object(EnvManager, "_get_default_env_paths")
    def test_config_set_empty_value(self, mock_default_paths, cli_runner, tmp_path):
        """Test setting a configuration key with empty value."""
        # Create a temporary .env.local file
        env_file = tmp_path / ".env.local"

        # Mock the default paths to use our temporary directory
        mock_default_paths.return_value = [env_file]

        # We need to re-initialize the EnvManager to use our mocked paths
        from local_coding_assistant.cli.commands import config

        config._env_manager = EnvManager(env_paths=[env_file])

        result = cli_runner.invoke(app, ["config", "set", "EMPTY_KEY", ""])

        assert result.exit_code == 0

        # The actual implementation doesn't output anything on success for set command
        assert result.stdout.strip() == ""

        # Verify the environment variable was set to empty string
        assert os.environ.get("LOCCA_EMPTY_KEY") == ""

        # Verify the value was written to the .env.local file
        assert env_file.exists()
        with open(env_file) as f:
            content = f.read()
            assert "LOCCA_EMPTY_KEY=" in content

    @patch.object(EnvManager, "_get_default_env_paths")
    def test_config_workflow(self, mock_default_paths, cli_runner, tmp_path):
        """Test a complete config workflow: set, get, verify."""
        # Create a temporary .env.local file
        env_file = tmp_path / ".env.local"

        # Mock the default paths to use our temporary directory
        mock_default_paths.return_value = [env_file]

        # We need to re-initialize the EnvManager to use our mocked paths
        from local_coding_assistant.cli.commands import config

        config._env_manager = EnvManager(env_paths=[env_file])

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
