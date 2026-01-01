"""
Unit tests for CLI provider management commands.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
import yaml
from typer.testing import CliRunner

from local_coding_assistant.cli.commands.provider import (
    _get_config_path,
    _load_config,
    _save_config,
    app,
    list_providers,
    validate,
)
from local_coding_assistant.config import EnvManager

# Create a test runner for the CLI
runner = CliRunner()


class TestProviderConfigHelpers:
    """Test provider configuration helper functions."""

    def test_get_config_path_default(self, test_configs):
        """Test getting default config path."""
        # Create a test env manager with the test path manager
        env_manager = EnvManager.create(
            path_manager=test_configs["path_manager"], load_env=False
        )

        # Get the config path using the test environment
        config_path = _get_config_path(env_manager=env_manager)

        # The path should be in the test config directory
        expected = test_configs["config_dir"] / "providers.local.yaml"
        assert str(config_path) == str(expected)

    @pytest.fixture
    def project_root(self):
        """Return the project root as a Path."""
        return Path(__file__).resolve().parents[3]

    def test_get_config_path_dev_mode(self, test_configs):
        """Test getting config path in dev mode."""
        # Create a test env manager with the test path manager
        env_manager = EnvManager.create(
            path_manager=test_configs["path_manager"], load_env=False
        )

        # Get the config path in dev mode
        config_path = _get_config_path(env_manager=env_manager)

        # The path should be in the test config directory
        expected = test_configs["config_dir"] / "providers.local.yaml"
        assert str(config_path) == str(expected)

    def test_load_config_invalid_yaml(self, test_configs):
        """Test loading config from invalid YAML file."""
        # Create an invalid YAML file in the test config directory
        config_path = test_configs["config_dir"] / "invalid_config.yaml"
        config_path.write_text("invalid: yaml: file: [")

        # Test that loading the invalid YAML raises an error
        with pytest.raises(yaml.YAMLError):
            _load_config(config_path)

    def test_save_config_invalid_data(self, test_configs):
        """Test saving config with unserializable data."""
        config_path = test_configs["config_dir"] / "test_invalid_data.yaml"

        # Create a custom class without proper YAML serialization
        class Unserializable:
            def __init__(self, value):
                self.value = value

            def __str__(self):
                return f"Unserializable({self.value})"

        test_obj = Unserializable("test")
        # The config will be wrapped in a 'providers' dict by _save_config
        test_config = {"test": test_obj}

        # This will serialize the object using PyYAML's default object representation
        _save_config(config_path, test_config)

        # Verify the content was serialized with PyYAML's object representation
        content = config_path.read_text()
        assert "!!python/object" in content
        assert "value: test" in content


class TestProviderAddCommand:
    """Test provider add command."""

    @pytest.fixture(autouse=True)
    def setup_method(self, test_configs):
        """Setup test environment for each test method."""
        self.test_configs = test_configs
        self.config_path = test_configs["config_dir"] / "providers.local.yaml"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # Create a test environment with API key
        self.original_env = os.environ.copy()
        os.environ["TEST_API_KEY"] = "env_test_key"

        yield

        # Cleanup
        os.environ.clear()
        os.environ.update(self.original_env)
        if self.config_path.exists():
            self.config_path.unlink()

    def extract_side_effect(self, value, default=None):
        """Helper function to mock _extract_value."""
        return value

    def verify_side_effect(self, llm_manager, provider_name):
        """Helper function to mock _verify_provider_health."""
        return True

    @patch("local_coding_assistant.cli.commands.provider.bootstrap")
    @patch("local_coding_assistant.cli.commands.provider.typer.echo")
    @patch("local_coding_assistant.cli.commands.provider._save_config")
    @patch("local_coding_assistant.cli.commands.provider._get_config_path")
    @patch("local_coding_assistant.cli.commands.provider._extract_value")
    @patch("local_coding_assistant.cli.commands.provider._verify_provider_health")
    def test_add_provider_minimal(
        self,
        mock_verify,
        mock_extract_value,
        mock_get_config_path,
        mock_save_config,
        mock_echo,
        mock_bootstrap,
    ):
        """Test adding provider with minimal configuration."""
        # Setup mocks
        mock_llm = MagicMock()
        mock_bootstrap.return_value = {"llm": mock_llm}

        mock_get_config_path.return_value = self.config_path
        mock_extract_value.side_effect = self.extract_side_effect
        mock_verify.return_value = True

        # Ensure the config directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # Call the function via CLI
        result = runner.invoke(
            app,
            [
                "add",
                "test_provider",
                "--base-url",
                "https://api.test.com",
                "--api-key-env",
                "TEST_API_KEY",
                "model1",
                "model2",
            ],
        )

        # Verify the result
        assert result.exit_code == 0
        mock_get_config_path.assert_called_once()
        mock_save_config.assert_called_once()
        mock_verify.assert_called_once()
        mock_echo.assert_called()

        # Verify _save_config was called with the correct arguments
        args, kwargs = mock_save_config.call_args
        assert kwargs["config_path"] == self.config_path
        assert kwargs["provider_name"] == "test_provider"

        # Verify the provider config
        provider_config = kwargs["config"]["providers"]["test_provider"]
        assert provider_config["base_url"] == "https://api.test.com"

    @patch("local_coding_assistant.cli.commands.provider.bootstrap")
    @patch("local_coding_assistant.cli.commands.provider.typer.echo")
    @patch("local_coding_assistant.cli.commands.provider._save_config")
    @patch("local_coding_assistant.cli.commands.provider._get_config_path")
    @patch("local_coding_assistant.cli.commands.provider._extract_value")
    @patch("local_coding_assistant.cli.commands.provider._verify_provider_health")
    def test_add_provider_with_all_options(
        self,
        mock_verify,
        mock_extract_value,
        mock_get_config_path,
        mock_save_config,
        mock_echo,
        mock_bootstrap,
    ):
        """Test adding provider with all optional parameters via CLI."""
        # Setup mocks
        mock_llm = MagicMock()
        mock_llm.reload_providers = MagicMock()
        mock_bootstrap.return_value = {"llm": mock_llm}

        mock_get_config_path.return_value = self.config_path
        mock_extract_value.side_effect = self.extract_side_effect
        mock_verify.return_value = True

        # Call the function via CLI with all options
        result = runner.invoke(
            app,
            [
                "add",
                "test_provider",
                "--base-url",
                "https://api.test.com",
                "--api-key",
                "test_key",
                "--api-key-env",
                "TEST_API_KEY",
                "--driver",
                "custom_driver",
                "--health-check-endpoint",
                "/health",
                "--log-level",
                "DEBUG",
                "model1",
                "model2",
            ],
        )

        # Verify the result
        assert result.exit_code == 0
        mock_get_config_path.assert_called_once()
        mock_save_config.assert_called_once()
        mock_verify.assert_called_once()
        mock_echo.assert_called()

        # Verify _save_config was called with correct parameters
        args, kwargs = mock_save_config.call_args
        assert kwargs["config_path"] == self.config_path
        assert kwargs["provider_name"] == "test_provider"

        # Verify the config contains all the provided options
        config = kwargs["config"]["providers"]["test_provider"]
        assert config["base_url"] == "https://api.test.com"
        assert config.get("driver") == "custom_driver"
        assert config.get("health_check_endpoint") == "/health"
        assert "model1" in config.get("models", [])
        assert "model2" in config.get("models", [])

    @patch.dict("os.environ", {"TEST_API_KEY": "env_test_key"}, clear=True)
    @patch("local_coding_assistant.cli.commands.provider.bootstrap")
    @patch("local_coding_assistant.cli.commands.provider.typer.echo")
    @patch("local_coding_assistant.cli.commands.provider._save_config")
    @patch("local_coding_assistant.cli.commands.provider._get_config_path")
    @patch("local_coding_assistant.cli.commands.provider._extract_value")
    @patch("local_coding_assistant.cli.commands.provider._verify_provider_health")
    def test_add_provider_with_health_check(
        self,
        mock_verify,
        mock_extract_value,
        mock_get_config_path,
        mock_save_config,
        mock_echo,
        mock_bootstrap,
    ):
        """Test adding provider with health check endpoint."""
        # Setup mocks
        mock_llm = MagicMock()
        mock_llm.reload_providers = MagicMock()
        mock_bootstrap.return_value = {"llm": mock_llm}

        mock_get_config_path.return_value = self.config_path
        mock_extract_value.side_effect = self.extract_side_effect
        mock_verify.return_value = True

        # Call the function via CLI with health check endpoint
        result = runner.invoke(
            app,
            [
                "add",
                "test_provider",
                "--base-url",
                "https://api.test.com",
                "--api-key-env",
                "TEST_API_KEY",
                "--health-check-endpoint",
                "/health",
                "model1",
            ],
        )

        # Verify the result
        assert result.exit_code == 0
        mock_get_config_path.assert_called_once()
        mock_save_config.assert_called_once()
        mock_verify.assert_called_once()
        mock_echo.assert_called()

        # Get the actual arguments passed to _save_config
        args, kwargs = mock_save_config.call_args

        # Verify the arguments
        assert kwargs["config_path"] == self.config_path
        assert kwargs["provider_name"] == "test_provider"
        assert "config" in kwargs
        assert "providers" in kwargs["config"]
        assert "test_provider" in kwargs["config"]["providers"]

        # Verify the provider config
        provider_config = kwargs["config"]["providers"]["test_provider"]
        assert provider_config["base_url"] == "https://api.test.com"
        assert provider_config["health_check_endpoint"] == "/health"
        assert provider_config.get("driver") == "openai_chat"  # Default driver

        # Verify _verify_provider_health was called with the correct arguments
        mock_verify.assert_called_once()
        # Check that the second argument is the provider name
        assert len(mock_verify.call_args[0]) >= 2
        assert mock_verify.call_args[0][1] == "test_provider"


class TestProviderListCommand:
    """Test provider list command."""

    @patch("local_coding_assistant.cli.commands.provider.bootstrap")
    @patch("local_coding_assistant.cli.commands.provider.typer.echo")
    def test_list_all_providers(self, mock_echo, mock_bootstrap):
        """Test listing all providers."""
        # Setup mock LLM manager with providers
        mock_llm = MagicMock()
        mock_llm.providers = {"openai": MagicMock(), "anthropic": MagicMock()}
        mock_bootstrap.return_value = {"llm": mock_llm}

        # Call the function
        list_providers()

        # Verify output
        assert mock_echo.call_count >= 2  # At least header and one provider

    @patch("local_coding_assistant.cli.commands.provider.bootstrap")
    @patch("local_coding_assistant.cli.commands.provider.typer.echo")
    def test_list_no_providers_configured(self, mock_echo, mock_bootstrap):
        """Test listing when no providers are configured."""
        # Setup mock LLM manager with no providers
        mock_llm = MagicMock()
        mock_llm.providers = {}
        mock_bootstrap.return_value = {"llm": mock_llm}

        # Call the function
        list_providers()

        # Verify the header is shown
        mock_echo.assert_any_call("Available providers:")
        mock_echo.assert_any_call("Name                 Source     Status       Models")
        mock_echo.assert_any_call(
            "------------------------------------------------------------"
        )

    @patch("local_coding_assistant.cli.commands.provider.bootstrap")
    @patch("local_coding_assistant.cli.commands.provider.typer.echo")
    def test_list_models_no_models_configured(self, mock_echo, mock_bootstrap):
        """Test listing models when provider has no models."""
        # Setup mock LLM manager with provider but no models
        mock_provider = MagicMock()
        mock_provider.get_available_models.return_value = []
        mock_provider_manager = MagicMock()
        mock_provider_manager.list_providers.return_value = ["test_provider"]
        mock_provider_manager.get_provider.return_value = mock_provider
        mock_provider_manager.get_provider_source.return_value = "test_source"

        mock_llm = MagicMock()
        mock_llm.provider_manager = mock_provider_manager
        mock_bootstrap.return_value = {"llm": mock_llm}

        # Call the function
        list_providers(provider="test_provider")

        # Verify appropriate messages are shown
        mock_echo.assert_any_call("Provider: test_provider (test_source)")
        mock_echo.assert_any_call("No models available")


class TestProviderRemoveCommand:
    """Test provider remove command."""

    @pytest.fixture(autouse=True)
    def setup_method(self, test_configs):
        """Setup test environment for each test method."""
        self.test_configs = test_configs
        self.config_path = test_configs["config_dir"] / "providers.local.yaml"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        yield
        if self.config_path.exists():
            self.config_path.unlink()

    @patch("local_coding_assistant.cli.commands.provider.bootstrap")
    @patch("local_coding_assistant.cli.commands.provider.typer.echo")
    @patch("local_coding_assistant.cli.commands.provider._save_config")
    @patch("local_coding_assistant.cli.commands.provider._load_config")
    @patch("local_coding_assistant.cli.commands.provider._get_config_path")
    @patch("local_coding_assistant.cli.commands.provider._verify_provider_health")
    def test_remove_existing_provider(
        self,
        mock_verify,
        mock_get_config_path,
        mock_load_config,
        mock_save_config,
        mock_echo,
        mock_bootstrap,
    ):
        """Test removing an existing provider."""
        # Setup mocks
        mock_llm = MagicMock()
        mock_llm.reload_providers = MagicMock()

        # Configure get_provider to raise an exception when called with "test_provider"
        def get_provider_side_effect(provider_name):
            if provider_name == "test_provider":
                raise ValueError(f"Provider '{provider_name}' not found")
            return MagicMock()  # Return a mock for other provider names

        mock_llm.get_provider.side_effect = get_provider_side_effect
        mock_llm.get_provider_status_list.return_value = [
            {"name": "another_provider", "status": "available"}
        ]
        mock_bootstrap.return_value = {"llm": mock_llm}

        # Setup verify mock to call reload_providers
        def verify_side_effect(llm_manager, provider_name):
            llm_manager.reload_providers()
            return None

        mock_verify.side_effect = verify_side_effect

        # Mock config with existing provider
        test_config = {
            "providers": {
                "test_provider": {"driver": "openai_chat"},
                "another_provider": {"driver": "custom"},
            }
        }
        mock_load_config.return_value = test_config
        mock_get_config_path.return_value = self.config_path

        # Call remove function via CLI
        result = runner.invoke(
            app,
            ["remove", "test_provider"],
        )

        # Verify the result
        assert result.exit_code == 0

        # Verify the provider was removed from config
        updated_config = mock_save_config.call_args[0][
            1
        ]  # Second argument to _save_config
        assert "test_provider" not in updated_config.get("providers", {})
        assert "another_provider" in updated_config.get("providers", {})

        # Verify the success message was printed
        mock_echo.assert_any_call("✅ Successfully removed provider 'test_provider'")


class TestProviderValidateCommand:
    """Test provider validate command."""

    @patch("local_coding_assistant.cli.commands.provider.typer.echo")
    @patch("local_coding_assistant.cli.commands.provider._load_config")
    @patch("local_coding_assistant.cli.commands.provider._get_config_path")
    def test_validate_valid_config(
        self, mock_get_config_path, mock_load_config, mock_echo
    ):
        """Test validating a valid provider configuration."""
        # Mock a valid config
        valid_config = {
            "providers": {
                "valid_provider": {
                    "driver": "openai_chat",
                    "base_url": "https://api.example.com",
                    "api_key": "test-api-key",  # Add api_key to avoid warnings
                    "models": {
                        "gpt-4": {"supported_parameters": ["temperature", "max_tokens"]}
                    },
                }
            }
        }
        mock_load_config.return_value = valid_config

        # Call validate function
        validate()

        # Verify success message
        mock_echo.assert_any_call("✅ Configuration is valid")

    @patch("local_coding_assistant.cli.commands.provider.typer.echo")
    @patch("local_coding_assistant.cli.commands.provider._load_config")
    @patch("local_coding_assistant.cli.commands.provider._get_config_path")
    def test_validate_invalid_config_missing_required_fields(
        self, mock_get_config_path, mock_load_config, mock_echo
    ):
        """Test validating a config with missing required fields."""
        # Mock an invalid config (missing base_url)
        invalid_config = {
            "providers": {
                "invalid_provider": {
                    "driver": "openai_chat"
                    # Missing required base_url
                }
            }
        }
        mock_load_config.return_value = invalid_config

        # Call validate function
        with pytest.raises(typer.Exit):
            validate()

        # Verify error message about missing field
        mock_echo.assert_any_call("❌ Error: Missing required fields: ['base_url']")

    @patch("local_coding_assistant.cli.commands.provider.typer.echo")
    @patch("local_coding_assistant.cli.commands.provider._load_config")
    @patch("local_coding_assistant.cli.commands.provider._get_config_path")
    def test_validate_config_with_warnings(
        self, mock_get_config_path, mock_load_config, mock_echo
    ):
        """Test validating a config with warnings but no errors."""
        # Mock a config with a warning (empty models list)
        warning_config = {
            "providers": {
                "warning_provider": {
                    "driver": "openai_chat",
                    "base_url": "https://api.example.com",
                    "models": {},
                }
            }
        }
        mock_load_config.return_value = warning_config

        # Call validate function
        validate()

        # Verify warning message and validation summary
        mock_echo.assert_any_call("⚠️  Warning: No models configured for this provider")
        mock_echo.assert_any_call("⚠️  Validation completed with warnings")

    @patch("local_coding_assistant.cli.commands.provider.typer.echo")
    @patch("local_coding_assistant.cli.commands.provider._load_config")
    @patch("local_coding_assistant.cli.commands.provider._get_config_path")
    def test_validate_nonexistent_config_file(
        self, mock_get_config_path, mock_load_config, mock_echo
    ):
        """Test validating a non-existent config file."""
        # Mock file not found
        mock_load_config.return_value = {}

        # Call validate function
        validate()

        # Verify appropriate message is shown for empty config
        mock_echo.assert_any_call("Configuration is empty")
