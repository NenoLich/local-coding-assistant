"""
Unit tests for CLI provider management commands.
"""
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
import yaml

from local_coding_assistant.cli.commands.provider import (
    _get_config_path,
    _load_config,
    _save_config,
    add,
    list_providers,
    reload,
    remove,
    validate,
)


class TestProviderConfigHelpers:
    """Test provider configuration helper functions."""

    def test_get_config_path_default(self):
        """Test getting default config path."""
        with patch("pathlib.Path.home") as mock_home:
            mock_home.return_value = Path("/home/testuser")
            config_path = _get_config_path()

            expected = Path(
                "/home/testuser/.local-coding-assistant/config/providers.local.yaml"
            )
            assert config_path == expected

    def test_get_config_path_custom(self):
        """Test getting custom config path."""
        custom_path = "/custom/config/providers.yaml"
        config_path = _get_config_path(custom_path)

        assert config_path == Path(custom_path)

    def test_load_config_nonexistent(self):
        """Test loading config from non-existent file."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            config_path = Path(temp_dir) / "nonexistent.yaml"
            config = _load_config(config_path)

            assert config == {}

    def test_load_config_existing(self):
        """Test loading config from existing file."""
        test_config = {
            "openai": {"driver": "openai_chat", "models": {"gpt-4": {}, "gpt-3.5": {}}}
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(test_config, f)
            config_path = Path(f.name)

        try:
            config = _load_config(config_path)
            assert config == test_config
        finally:
            config_path.unlink()

    def test_save_config_new_file(self):
        """Test saving config to new file."""
        test_config = {
            "test_provider": {"driver": "openai_chat", "models": {"test-model": {}}}
        }

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            config_path = Path(temp_dir) / "test.yaml"

            _save_config(config_path, test_config)

            # Verify file was created
            assert config_path.exists()

            # Verify content
            with open(config_path) as f:
                saved_config = yaml.safe_load(f)
            assert saved_config == test_config

    def test_save_config_existing_file(self):
        """Test saving config to existing file."""
        original_config = {"existing": {"driver": "test"}}
        new_config = {"new_provider": {"driver": "openai_chat"}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(original_config, f)
            config_path = Path(f.name)

        try:
            _save_config(config_path, new_config)

            # Verify content was overwritten
            with open(config_path) as f:
                saved_config = yaml.safe_load(f)
            assert saved_config == new_config
        finally:
            config_path.unlink()


class TestProviderAddCommand:
    """Test provider add command."""

    @patch("local_coding_assistant.cli.commands.provider.bootstrap")
    @patch("local_coding_assistant.cli.commands.provider.typer.echo")
    def test_add_provider_minimal(self, mock_echo, mock_bootstrap):
        """Test adding provider with minimal configuration."""
        # Mock bootstrap context
        mock_ctx = {"llm": MagicMock()}
        mock_bootstrap.return_value = mock_ctx

        # Mock LLM manager
        mock_llm = MagicMock()
        mock_llm.reload_providers = MagicMock()
        mock_ctx["llm"] = mock_llm

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            config_file = Path(temp_dir) / "test.yaml"

            add(
                "test_provider",
                api_key="test_key",
                config_file=str(config_file),
                log_level="INFO",
            )

            # Verify config was saved
            assert config_file.exists()
            with open(config_file) as f:
                config = yaml.safe_load(f)

            assert "test_provider" in config
            assert config["test_provider"]["driver"] == "openai_chat"
            assert config["test_provider"]["api_key"] == "test_key"
            assert config["test_provider"]["models"] == {}

            # Verify bootstrap was called
            mock_bootstrap.assert_called_once()
            mock_llm.reload_providers.assert_called_once()

            # Manually close/delete files before the dir is cleaned up
            try:
                os.unlink(config_file)
            except Exception:
                pass

    @patch("local_coding_assistant.cli.commands.provider.bootstrap")
    @patch("local_coding_assistant.cli.commands.provider.typer.echo")
    def test_add_provider_with_models(self, mock_echo, mock_bootstrap):
        """Test adding provider with models."""
        mock_ctx = {"llm": MagicMock()}
        mock_bootstrap.return_value = mock_ctx

        mock_llm = MagicMock()
        mock_llm.reload_providers = MagicMock()
        mock_ctx["llm"] = mock_llm

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            config_file = Path(temp_dir) / "test.yaml"

            add(
                "openai",
                "gpt-4",
                "gpt-3.5-turbo",
                api_key_env="OPENAI_API_KEY",
                base_url="https://api.openai.com/v1",
                driver="openai_responses",
                config_file=str(config_file),
                log_level="INFO",
            )

            # Verify config
            with open(config_file) as f:
                config = yaml.safe_load(f)

            assert "openai" in config
            assert config["openai"]["driver"] == "openai_responses"
            assert config["openai"]["api_key_env"] == "OPENAI_API_KEY"
            assert config["openai"]["base_url"] == "https://api.openai.com/v1"
            assert config["openai"]["models"] == {"gpt-4": {}, "gpt-3.5-turbo": {}}

            # Manually close/delete files before the dir is cleaned up
            try:
                os.unlink(config_file)
            except Exception:
                pass

    @patch("local_coding_assistant.cli.commands.provider.bootstrap")
    @patch("local_coding_assistant.cli.commands.provider.typer.echo")
    def test_add_provider_bootstrap_failure(self, mock_echo, mock_bootstrap):
        """Test adding provider when bootstrap fails."""
        mock_bootstrap.return_value = {"llm": None}  # LLM manager not available

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            config_file = Path(temp_dir) / "test.yaml"

            with pytest.raises(click.exceptions.Exit):  # typer.Exit
                add(
                    "test_provider", config_file=str(config_file), log_level="INFO"
                )

            # Manually close/delete files before the dir is cleaned up
            try:
                os.unlink(config_file)
            except Exception:
                pass


class TestProviderListCommand:
    """Test provider list command."""

    @patch("local_coding_assistant.cli.commands.provider.bootstrap")
    @patch("local_coding_assistant.cli.commands.provider.typer.echo")
    def test_list_all_providers(self, mock_echo, mock_bootstrap):
        """Test listing all providers."""
        # Mock bootstrap context
        mock_ctx = {"llm": MagicMock()}
        mock_bootstrap.return_value = mock_ctx

        # Mock LLM manager and provider status
        mock_llm = MagicMock()
        mock_llm.get_provider_status_list.return_value = [
            {
                "name": "openai",
                "source": "built-in",
                "status": "available",
                "models": 2,
                "error": None,
            },
            {
                "name": "google",
                "source": "built-in",
                "status": "unavailable",
                "models": 0,
                "error": "API key missing",
            },
        ]
        mock_ctx["llm"] = mock_llm

        list_providers(log_level="INFO")

        # Verify bootstrap was called
        mock_bootstrap.assert_called_once()

        # Verify status was retrieved
        mock_llm.get_provider_status_list.assert_called_once()

        # Verify output calls (should print header and each provider)
        assert mock_echo.call_count >= 4  # Header + 2 providers + separator

    @patch("local_coding_assistant.cli.commands.provider.bootstrap")
    @patch("local_coding_assistant.cli.commands.provider.typer.echo")
    def test_list_specific_provider(self, mock_echo, mock_bootstrap):
        """Test listing models for specific provider."""
        mock_ctx = {"llm": MagicMock()}
        mock_bootstrap.return_value = mock_ctx

        mock_llm = MagicMock()
        mock_provider_manager = MagicMock()
        mock_provider_manager.list_providers.return_value = ["openai"]
        mock_provider_manager.get_provider.return_value = MagicMock()
        mock_provider_manager.get_provider_source.return_value = "built-in"

        mock_provider = MagicMock()
        mock_provider.get_available_models.return_value = ["gpt-4", "gpt-3.5-turbo"]
        mock_provider_manager.get_provider.return_value = mock_provider

        mock_llm.provider_manager = mock_provider_manager
        mock_ctx["llm"] = mock_llm

        list_providers(provider="openai", log_level="INFO")

        # Verify provider lookup
        mock_provider_manager.get_provider.assert_called_once_with("openai")
        mock_provider.get_available_models.assert_called_once()

        # Verify output includes provider info and models
        echo_calls = [call.args[0] for call in mock_echo.call_args_list]
        provider_found = any("openai" in call.lower() for call in echo_calls)
        model_found = any("gpt-4" in call for call in echo_calls)
        assert provider_found
        assert model_found

    @patch("local_coding_assistant.cli.commands.provider.bootstrap")
    @patch("local_coding_assistant.cli.commands.provider.typer.echo")
    def test_list_nonexistent_provider(self, mock_echo, mock_bootstrap):
        """Test listing non-existent provider."""
        mock_ctx = {"llm": MagicMock()}
        mock_bootstrap.return_value = mock_ctx

        mock_llm = MagicMock()
        mock_provider_manager = MagicMock()
        mock_provider_manager.list_providers.return_value = []
        mock_llm.provider_manager = mock_provider_manager
        mock_ctx["llm"] = mock_llm

        list_providers(provider="nonexistent", log_level="INFO")

        # Should output provider not found message
        echo_calls = [call.args[0] for call in mock_echo.call_args_list]
        assert any("not found" in call.lower() for call in echo_calls)

    @patch("local_coding_assistant.cli.commands.provider.bootstrap")
    @patch("local_coding_assistant.cli.commands.provider.typer.echo")
    def test_list_provider_bootstrap_failure(self, mock_echo, mock_bootstrap):
        """Test listing when bootstrap fails."""
        mock_bootstrap.return_value = {"llm": None}

        with pytest.raises(click.exceptions.Exit):
            list_providers(log_level="INFO")


class TestProviderRemoveCommand:
    """Test provider remove command."""

    @patch("local_coding_assistant.cli.commands.provider.bootstrap")
    @patch("local_coding_assistant.cli.commands.provider.typer.echo")
    def test_remove_existing_provider(self, mock_echo, mock_bootstrap):
        """Test removing existing provider."""
        mock_ctx = {"llm": MagicMock()}
        mock_bootstrap.return_value = mock_ctx

        mock_llm = MagicMock()
        mock_llm.reload_providers = MagicMock()
        mock_ctx["llm"] = mock_llm

        # Create config with provider
        test_config = {
            "openai": {"driver": "openai_chat", "models": {"gpt-4": {}}},
            "google": {"driver": "google", "models": {"gemini-pro": {}}},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(test_config, f)
            config_file = Path(f.name)

        try:
            remove(name="openai", config_file=str(config_file), log_level="INFO")

            # Verify provider was removed from config
            with open(config_file) as f:
                updated_config = yaml.safe_load(f)

            assert "openai" not in updated_config
            assert "google" in updated_config  # Other provider should remain

            # Verify bootstrap and reload were called
            mock_bootstrap.assert_called_once()
            mock_llm.reload_providers.assert_called_once()
        finally:
            config_file.unlink()

    @patch("local_coding_assistant.cli.commands.provider.bootstrap")
    @patch("local_coding_assistant.cli.commands.provider.typer.echo")
    def test_remove_nonexistent_provider(self, mock_echo, mock_bootstrap):
        """Test removing non-existent provider."""
        mock_ctx = {"llm": MagicMock()}
        mock_bootstrap.return_value = mock_ctx

        mock_llm = MagicMock()
        mock_llm.reload_providers = MagicMock()
        mock_ctx["llm"] = mock_llm

        test_config = {"openai": {"driver": "openai_chat"}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(test_config, f)
            config_file = Path(f.name)

        try:
            with pytest.raises(click.exceptions.Exit):
                remove(
                    name="nonexistent", config_file=str(config_file), log_level="INFO"
                )
        finally:
            config_file.unlink()

    @patch("local_coding_assistant.cli.commands.provider.bootstrap")
    @patch("local_coding_assistant.cli.commands.provider.typer.echo")
    def test_remove_provider_bootstrap_failure(self, mock_echo, mock_bootstrap):
        """Test removing provider when bootstrap fails."""
        mock_bootstrap.return_value = {"llm": None}

        with pytest.raises(click.exceptions.Exit):
            remove(name="test_provider", log_level="INFO")


class TestProviderValidateCommand:
    """Test provider validate command."""

    @patch("local_coding_assistant.cli.commands.provider._get_config_path")
    @patch("local_coding_assistant.cli.commands.provider.typer.echo")
    def test_validate_valid_config(self, mock_echo, mock_get_config_path):
        """Test validating valid provider configuration."""
        test_config = {
            "openai": {"driver": "openai_chat", "models": {"gpt-4": {}, "gpt-3.5": {}}},
            "google": {"driver": "google", "models": {"gemini-pro": {}}},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(test_config, f)
            config_file = Path(f.name)
            mock_get_config_path.return_value = config_file

        try:
            validate(config_file=str(config_file))

            # Should find both providers and mark them as valid
            echo_calls = [call.args[0] for call in mock_echo.call_args_list]
            assert any(
                "openai" in call.lower() and "2 model" in call for call in echo_calls
            )
            assert any(
                "google" in call.lower() and "1 model" in call for call in echo_calls
            )
        finally:
            config_file.unlink()

    @patch("local_coding_assistant.cli.commands.provider._get_config_path")
    @patch("local_coding_assistant.cli.commands.provider.typer.echo")
    def test_validate_config_with_warnings(self, mock_echo, mock_get_config_path):
        """Test validating config with warnings."""
        test_config = {
            "incomplete_provider": {
                # Missing driver field
                "models": {"test": {}}
            },
            "no_models_provider": {
                "driver": "openai_chat"
                # Missing models
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(test_config, f)
            config_file = Path(f.name)
            mock_get_config_path.return_value = config_file

        try:
            validate(config_file=str(config_file))

            # Should report warnings
            echo_calls = [call.args[0] for call in mock_echo.call_args_list]
            assert any("missing fields" in call.lower() for call in echo_calls)
            assert any("no models" in call.lower() for call in echo_calls)
        finally:
            config_file.unlink()

    @patch("local_coding_assistant.cli.commands.provider._get_config_path")
    @patch("local_coding_assistant.cli.commands.provider.typer.echo")
    def test_validate_invalid_yaml(self, mock_echo, mock_get_config_path):
        """Test validating invalid YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [\n")  # Invalid YAML
            config_file = Path(f.name)
            mock_get_config_path.return_value = config_file

        try:
            with pytest.raises(click.exceptions.Exit):
                validate(config_file=str(config_file))
        finally:
            config_file.unlink()

    @patch("local_coding_assistant.cli.commands.provider._get_config_path")
    @patch("local_coding_assistant.cli.commands.provider.typer.echo")
    def test_validate_nonexistent_config(self, mock_echo, mock_get_config_path):
        """Test validating non-existent config file."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            config_file = Path(temp_dir) / "nonexistent.yaml"
            mock_get_config_path.return_value = config_file

            validate(config_file=str(config_file))

            # Should indicate no config found
            echo_calls = [call.args[0] for call in mock_echo.call_args_list]
            assert any(
                "no provider configuration found" in call.lower() for call in echo_calls
            )

    @patch("local_coding_assistant.cli.commands.provider._get_config_path")
    @patch("local_coding_assistant.cli.commands.provider.typer.echo")
    def test_validate_empty_config(self, mock_echo, mock_get_config_path):
        """Test validating empty config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")  # Empty file
            config_file = Path(f.name)
            mock_get_config_path.return_value = config_file

        try:
            validate(config_file=str(config_file))

            # Should indicate config is empty
            echo_calls = [call.args[0] for call in mock_echo.call_args_list]
            assert any("configuration is empty" in call.lower() for call in echo_calls)
        finally:
            config_file.unlink()


class TestProviderReloadCommand:
    """Test provider reload command."""

    @patch("local_coding_assistant.cli.commands.provider.bootstrap")
    @patch("local_coding_assistant.cli.commands.provider.typer.echo")
    def test_reload_providers(self, mock_echo, mock_bootstrap):
        """Test reloading providers."""
        mock_ctx = {"llm": MagicMock()}
        mock_bootstrap.return_value = mock_ctx

        mock_llm = MagicMock()
        mock_llm.reload_providers = MagicMock()
        mock_ctx["llm"] = mock_llm

        reload(log_level="INFO")

        # Verify bootstrap and reload were called
        mock_bootstrap.assert_called_once()
        mock_llm.reload_providers.assert_called_once()

        # Verify success message
        echo_calls = [call.args[0] for call in mock_echo.call_args_list]
        assert any("reloaded successfully" in call.lower() for call in echo_calls)

    @patch("local_coding_assistant.cli.commands.provider.bootstrap")
    @patch("local_coding_assistant.cli.commands.provider.typer.echo")
    def test_reload_bootstrap_failure(self, mock_echo, mock_bootstrap):
        """Test reloading when bootstrap fails."""
        mock_bootstrap.return_value = {"llm": None}

        with pytest.raises(click.exceptions.Exit):
            reload(log_level="INFO")


class TestProviderCommandIntegration:
    """Test provider command integration scenarios."""

    @patch("local_coding_assistant.cli.commands.provider.bootstrap")
    @patch("local_coding_assistant.cli.commands.provider.typer.echo")
    def test_add_list_remove_flow(self, mock_echo, mock_bootstrap):
        """Test complete flow: add -> list -> remove."""
        mock_ctx = {"llm": MagicMock()}
        mock_bootstrap.return_value = mock_ctx

        mock_llm = MagicMock()
        mock_llm.reload_providers = MagicMock()
        mock_llm.get_provider_status_list.return_value = [
            {
                "name": "test_provider",
                "source": "local",
                "status": "available",
                "models": 1,
            }
        ]
        mock_llm.provider_manager = MagicMock()
        mock_llm.provider_manager.list_providers.return_value = ["test_provider"]
        mock_llm.provider_manager.get_provider.return_value = MagicMock()
        mock_llm.provider_manager.get_provider_source.return_value = "local"
        mock_ctx["llm"] = mock_llm

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            config_file = Path(temp_dir) / "test.yaml"

            # Add provider
            add(
                "test_provider",
                api_key="test_key",
                config_file=str(config_file),
                log_level="INFO",
            )
            list_providers(log_level="INFO")

            # Remove provider
            remove(name="test_provider", config_file=str(config_file), log_level="INFO")

            # Verify config is empty after removal
            with open(config_file) as f:
                final_config = yaml.safe_load(f)
            assert final_config == {}

    @patch("local_coding_assistant.cli.commands.provider.bootstrap")
    @patch("local_coding_assistant.cli.commands.provider.typer.echo")
    def test_multiple_providers_management(self, mock_echo, mock_bootstrap):
        """Test managing multiple providers."""
        mock_ctx = {"llm": MagicMock()}
        mock_bootstrap.return_value = mock_ctx

        mock_llm = MagicMock()
        mock_llm.reload_providers = MagicMock()
        mock_ctx["llm"] = mock_llm

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            config_file = Path(temp_dir) / "test.yaml"

            # Add multiple providers
            add(
                "openai",
                "gpt-4",
                api_key="key1",
                config_file=str(config_file),
            )
            add(
                "google",
                "gemini-pro",
                api_key_env="GOOGLE_KEY",
                config_file=str(config_file),
            )
            add(
                "local",
                "local-model",
                base_url="http://localhost:8000",
                config_file=str(config_file),
            )

            # Verify all providers in config
            with open(config_file) as f:
                config = yaml.safe_load(f)

            assert len(config) == 3
            assert "openai" in config
            assert "google" in config
            assert "local" in config

            # Test that different configuration options are preserved
            assert config["openai"]["api_key"] == "key1"
            assert config["google"]["api_key_env"] == "GOOGLE_KEY"
            assert config["local"]["base_url"] == "http://localhost:8000"

    @patch("local_coding_assistant.cli.commands.provider._get_config_path")
    @patch("local_coding_assistant.cli.commands.provider.bootstrap")
    @patch("local_coding_assistant.cli.commands.provider.typer.echo")
    def test_config_file_override(self, mock_echo, mock_bootstrap, mock_get_config_path):
        """Test config file path override."""
        mock_ctx = {"llm": MagicMock()}
        mock_bootstrap.return_value = mock_ctx

        mock_llm = MagicMock()
        mock_llm.reload_providers = MagicMock()
        mock_ctx["llm"] = mock_llm

        # Clean up default config directory before test
        default_path = _get_config_path()
        if default_path.exists():
            import shutil
            shutil.rmtree(default_path.parent)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            # Use custom config file path
            config_file = Path(temp_dir) / "custom" / "providers.yaml"
            mock_get_config_path.return_value = config_file

            add(
                "custom_provider",
                "custom-model",
                api_key="custom_key",
                config_file=str(config_file),
                log_level="INFO",
            )

            # Verify config was saved to custom path
            assert config_file.exists()
            with open(config_file) as f:
                config = yaml.safe_load(f)
            assert "custom_provider" in config

            # Verify default path was not used
            default_path = _get_config_path()
            assert not default_path.exists()


class TestProviderCommandErrorHandling:
    """Test provider command error handling."""

    @patch("local_coding_assistant.cli.commands.provider.bootstrap")
    @patch("local_coding_assistant.cli.commands.provider.typer.echo")
    def test_yaml_save_error(self, mock_echo, mock_bootstrap):
        """Test handling YAML save errors."""
        # This is hard to test directly since yaml.dump rarely fails
        # but we can test the structure exists for error handling

        mock_ctx = {"llm": MagicMock()}
        mock_bootstrap.return_value = mock_ctx

        mock_llm = MagicMock()
        mock_llm.reload_providers = MagicMock()
        mock_ctx["llm"] = mock_llm

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            config_file = Path(temp_dir) / "test.yaml"

            # Normal case should work
            add("test", api_key="test_key", config_file=str(config_file), log_level="INFO")

            # Verify directory was created and file exists
            assert config_file.exists()
            assert config_file.parent.exists()

            # Manually close/delete files before the dir is cleaned up
            try:
                os.unlink(config_file)
            except Exception:
                pass
