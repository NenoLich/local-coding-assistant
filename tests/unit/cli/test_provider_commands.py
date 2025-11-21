"""
Unit tests for CLI provider management commands.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
import yaml

from local_coding_assistant.cli.commands.provider import (
    _get_config_path,
    _load_config,
    _save_config,
    add,
    list_providers,
    remove,
    validate,
)


class TestProviderConfigHelpers:
    """Test provider configuration helper functions."""

    def test_get_config_path_default(self):
        """Test getting default config path."""
        with (
            patch("pathlib.Path.home") as mock_home,
            patch("os.getenv", return_value=None),
        ):  # Ensure dev mode is not active
            mock_home.return_value = Path("/home/testuser")
            config_path = _get_config_path()

            # The default path should be in the user's home directory
            expected = Path(
                "/home/testuser/.local-coding-assistant/config/providers.local.yaml"
            )
            assert str(config_path) == str(expected)

    @pytest.fixture
    def project_root(self):
        """Return the project root as a Path."""
        return Path(__file__).resolve().parents[3]

    def test_get_config_path_dev_mode(self, project_root):
        """Test getting config path in dev mode."""
        # Create the expected config file path
        expected = (
            project_root
            / "src"
            / "local_coding_assistant"
            / "config"
            / "providers.local.yaml"
        )

        # Call the function
        result = _get_config_path(dev=True)

        # Verify the result is a Path object
        assert isinstance(result, Path)

        # Verify the path is correct
        assert str(result) == str(expected)

    def test_load_config_invalid_yaml(self):
        """Test loading config from invalid YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: file: [")
            config_path = Path(f.name)

        try:
            with pytest.raises(yaml.YAMLError):
                _load_config(config_path)
        finally:
            if config_path.exists():
                config_path.unlink()

    def test_save_config_invalid_data(self):
        """Test saving config with unserializable data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test.yaml"

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
            with open(config_path) as f:
                content = f.read()
                # Check for the YAML object tag with our class path
                assert (
                    "!!python/object:tests.unit.cli.test_provider_commands.Unserializable"
                    in content
                )
                # Check that the value is preserved
                assert "value: test" in content


class TestProviderAddCommand:
    """Test provider add command."""

    @patch.dict("os.environ", {"TEST_API_KEY": "env_test_key"}, clear=True)
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
        mock_llm.get_provider_status_list.return_value = [
            {"name": "test_provider", "status": "available"}
        ]
        mock_llm.reload_providers = MagicMock()

        # Set up bootstrap to return our mock LLM manager
        mock_ctx = {"llm": mock_llm}
        mock_bootstrap.return_value = mock_ctx

        # Mock _extract_value to handle Typer parameters
        def extract_side_effect(value, default=None):
            if hasattr(value, "default"):
                return value.default
            return value if value is not None else default

        mock_extract_value.side_effect = extract_side_effect

        # Mock the models parameter to be properly handled as a list
        mock_models = MagicMock()
        mock_models.default = (
            None  # This will make _extract_value return None for models
        )
        mock_extract_value.return_value = None  # Return None for models by default

        # Create a temporary directory for the test
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            temp_dir_path = Path(temp_dir)
            config_file = temp_dir_path / "test.yaml"

            # Mock _get_config_path to return our test path
            mock_get_config_path.return_value = config_file

            # Mock the bootstrap function to return our mock LLM manager
            mock_bootstrap.return_value = {"llm": mock_llm}

            # Setup the verify mock to call reload_providers
            def verify_side_effect(llm_manager, provider_name):
                llm_manager.reload_providers()
                return None

            mock_verify.side_effect = verify_side_effect

            # Call the add function with proper Typer parameter handling
            add(
                "test_provider",  # name
                mock_models,  # models (as a mock)
                api_key="test_key",
                base_url="https://api.example.com",
                config_file=str(config_file),
                log_level="INFO",
            )

            # Verify _save_config was called once
            mock_save_config.assert_called_once()

            # Get the actual arguments passed to _save_config
            args, kwargs = mock_save_config.call_args

            # Verify the arguments
            assert kwargs["config_path"] == config_file
            assert kwargs["provider_name"] == "test_provider"
            assert "config" in kwargs
            assert "providers" in kwargs["config"]
            assert "test_provider" in kwargs["config"]["providers"]

            # Verify the provider config
            provider_config = kwargs["config"]["providers"]["test_provider"]
            assert provider_config["api_key"] == "test_key"
            assert provider_config["base_url"] == "https://api.example.com"
            assert provider_config["driver"] == "openai_chat"  # Default driver

            # Verify bootstrap was called
            mock_bootstrap.assert_called_once()

            # Verify _verify_provider_health was called with the correct arguments
            mock_verify.assert_called_once_with(mock_llm, "test_provider")

            # Verify reload_providers was called
            mock_llm.reload_providers.assert_called_once()

    @patch("local_coding_assistant.cli.commands.provider.bootstrap")
    @patch("local_coding_assistant.cli.commands.provider.typer.echo")
    @patch("local_coding_assistant.cli.commands.provider._save_config")
    @patch("local_coding_assistant.cli.commands.provider._get_config_path")
    @patch("local_coding_assistant.cli.commands.provider._verify_provider_health")
    def test_add_provider_with_all_options(
        self,
        mock_verify,
        mock_get_config_path,
        mock_save_config,
        mock_echo,
        mock_bootstrap,
    ):
        """Test adding provider with all optional parameters."""
        # Setup mocks
        mock_llm = MagicMock()
        mock_llm.reload_providers = MagicMock()
        mock_llm.get_provider_status_list.return_value = [
            {"name": "test_provider", "status": "available"}
        ]
        mock_bootstrap.return_value = {"llm": mock_llm}

        # Setup verify mock to call reload_providers
        def verify_side_effect(llm_manager, provider_name):
            llm_manager.reload_providers()
            return None

        mock_verify.side_effect = verify_side_effect

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as temp_file:
            config_path = Path(temp_file.name)
            mock_get_config_path.return_value = config_path

            # Call the add function with all options
            add(
                "test_provider",
                models=["model1", "model2"],
                api_key="test_key",
                api_key_env="TEST_API_KEY",
                base_url="https://api.example.com",
                dev=True,
                driver="custom_driver",
                health_check_endpoint="/health",
                config_file=str(config_path),
                log_level="DEBUG",
            )

            # Verify _save_config was called with correct parameters
            mock_save_config.assert_called_once()
            args, kwargs = mock_save_config.call_args

            # Verify the config contains all the provided options
            config = kwargs["config"]["providers"]["test_provider"]
            assert config["api_key"] == "test_key"
            assert config["api_key_env"] == "TEST_API_KEY"
            assert config["base_url"] == "https://api.example.com"
            assert config["driver"] == "custom_driver"
            assert config["health_check_endpoint"] == "/health"
            assert "model1" in config["models"]
            assert "model2" in config["models"]

            # Call the add function with API key from env
            add(
                "test_provider",
                base_url="https://api.example.com",
                api_key_env="TEST_API_KEY",
                config_file=str(config_path),
            )

            # Verify _save_config was called with the API key from env
            mock_save_config.assert_called_once()
            args, kwargs = mock_save_config.call_args

            # The config should use the API key from environment
            config = kwargs["config"]["providers"]["test_provider"]
            assert config["api_key"] == "test_key"
            assert config["api_key_env"] == "TEST_API_KEY"

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
        mock_llm.get_provider_status_list.return_value = [
            {"name": "test_provider", "status": "available"}
        ]
        mock_llm.reload_providers = MagicMock()

        # Set up bootstrap to return our mock LLM manager
        mock_ctx = {"llm": mock_llm}
        mock_bootstrap.return_value = mock_ctx

        # Mock _extract_value to handle Typer parameters
        def extract_side_effect(value, default=None):
            if hasattr(value, "default"):
                return value.default
            # For specific values we know will be passed
            if value == "INFO":
                return "INFO"
            if value == "openai_chat":
                return "openai_chat"
            if value == "/health":
                return "/health"
            if value == "https://api.example.com":
                return "https://api.example.com"
            return value if value is not None else default

        mock_extract_value.side_effect = extract_side_effect

        # Mock the models parameter to be properly handled as a list
        mock_models = MagicMock()
        mock_models.default = (
            None  # This will make _extract_value return None for models
        )
        mock_extract_value.return_value = None  # Return None for models by default

        # Create a temporary directory for the test
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            temp_dir_path = Path(temp_dir)
            config_file = temp_dir_path / "test.yaml"

            # Mock _get_config_path to return our test path
            mock_get_config_path.return_value = config_file

            # Setup the verify mock to call reload_providers
            def verify_side_effect(llm_manager, provider_name):
                llm_manager.reload_providers()
                return None

            mock_verify.side_effect = verify_side_effect

            # Call the add function with health check endpoint
            add(
                "test_provider",  # name
                mock_models,  # models (as a mock)
                base_url="https://api.example.com",
                health_check_endpoint="/health",
                config_file=str(config_file),
                log_level="INFO",
            )

            # Verify _save_config was called once
            mock_save_config.assert_called_once()

            # Get the actual arguments passed to _save_config
            args, kwargs = mock_save_config.call_args

            # Verify the arguments
            assert kwargs["config_path"] == config_file
            assert kwargs["provider_name"] == "test_provider"
            assert "config" in kwargs
            assert "providers" in kwargs["config"]
            assert "test_provider" in kwargs["config"]["providers"]

            # Verify the provider config
            provider_config = kwargs["config"]["providers"]["test_provider"]
            assert provider_config["base_url"] == "https://api.example.com"
            assert provider_config["health_check_endpoint"] == "/health"
            assert provider_config["driver"] == "openai_chat"  # Default driver

            # Verify bootstrap was called
            mock_bootstrap.assert_called_once()

            # Verify _verify_provider_health was called with the correct arguments
            mock_verify.assert_called_once_with(mock_llm, "test_provider")

            # Verify reload_providers was called
            mock_llm.reload_providers.assert_called_once()


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

        # Mock config path
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as temp_file:
            config_path = Path(temp_file.name)
            mock_get_config_path.return_value = config_path

            # Call remove function
            remove("test_provider", config_file=str(config_path))

        # Verify the provider was removed from config
        updated_config = mock_save_config.call_args[0][1]  # Second argument to _save_config
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
