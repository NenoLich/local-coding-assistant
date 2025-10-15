"""Unit tests for the configuration system."""

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from local_coding_assistant.config.loader import ConfigLoader
from local_coding_assistant.config.schemas import AppConfig, LLMConfig, RuntimeConfig
from local_coding_assistant.core.exceptions import ConfigError


class TestLLMConfig:
    """Test LLMConfig schema validation."""

    def test_llm_config_defaults(self):
        """Test LLMConfig with default values."""
        config = LLMConfig()
        assert config.provider == "openai"
        assert config.temperature == 0.7
        assert config.max_tokens is None
        assert config.api_key is None

    def test_llm_config_with_overrides(self):
        """Test LLMConfig with_overrides method."""
        config = LLMConfig(model_name="gpt-3.5-turbo", temperature=0.7)

        # Test single override
        overridden = config.with_overrides(model_name="gpt-4")
        assert overridden.model_name == "gpt-4"
        assert overridden.temperature == 0.7  # Unchanged

        # Test multiple overrides
        overridden = config.with_overrides(
            model_name="gpt-4", temperature=0.8, max_tokens=1000
        )
        assert overridden.model_name == "gpt-4"
        assert overridden.temperature == 0.8
        assert overridden.max_tokens == 1000

    def test_llm_config_with_invalid_overrides(self):
        """Test LLMConfig with_overrides method with invalid values."""
        config = LLMConfig(model_name="gpt-3.5-turbo", temperature=0.7)

        # Test invalid temperature
        with pytest.raises(ValueError, match="Invalid LLM configuration override"):
            config.with_overrides(temperature=-1)

        # Test invalid max_tokens
        with pytest.raises(ValueError, match="Invalid LLM configuration override"):
            config.with_overrides(max_tokens=0)

    def test_llm_config_with_none_overrides(self):
        """Test LLMConfig with_overrides method with None values (should be ignored)."""
        config = LLMConfig(model_name="gpt-3.5-turbo", temperature=0.7)

        # Test with None values - should be ignored
        overridden = config.with_overrides(
            model_name=None, temperature=0.8, max_tokens=None
        )

        # Assertions
        assert overridden.model_name == "gpt-3.5-turbo"  # Should remain unchanged
        assert overridden.temperature == 0.8  # Should be updated
        assert overridden.max_tokens is None  # Should remain unchanged
        assert overridden.provider == "openai"  # Should remain unchanged

    def test_llm_config_validation(self):
        """Test LLMConfig validation."""
        # Test temperature bounds
        with pytest.raises(ValueError):
            LLMConfig(temperature=-0.1)

        with pytest.raises(ValueError):
            LLMConfig(temperature=2.1)

        # Test max_tokens positive
        with pytest.raises(ValueError):
            LLMConfig(max_tokens=0)

        with pytest.raises(ValueError):
            LLMConfig(max_tokens=-1)


class TestRuntimeConfig:
    """Test RuntimeConfig schema validation."""

    def test_runtime_config_defaults(self):
        """Test RuntimeConfig with default values."""
        config = RuntimeConfig()
        assert config.persistent_sessions is False
        assert config.max_session_history == 100
        assert config.enable_logging is True
        assert config.log_level == "INFO"

    def test_runtime_config_custom_values(self):
        """Test RuntimeConfig with custom values."""
        config = RuntimeConfig(
            persistent_sessions=True,
            max_session_history=50,
            enable_logging=False,
            log_level="DEBUG",
        )
        assert config.persistent_sessions is True
        assert config.max_session_history == 50
        assert config.enable_logging is False
        assert config.log_level == "DEBUG"

    def test_runtime_config_validation(self):
        """Test RuntimeConfig validation."""
        # Test max_session_history positive
        with pytest.raises(ValueError):
            RuntimeConfig(max_session_history=0)

        with pytest.raises(ValueError):
            RuntimeConfig(max_session_history=-1)


class TestAppConfig:
    """Test AppConfig schema validation."""

    def test_app_config_defaults(self):
        """Test AppConfig with default values."""
        config = AppConfig()
        assert isinstance(config.llm, LLMConfig)
        assert isinstance(config.runtime, RuntimeConfig)
        assert config.llm.model_name == "gpt-5-mini"
        assert config.runtime.persistent_sessions is False

    def test_config_loader_with_env_loader(self):
        """Test ConfigLoader integration with EnvLoader."""
        from local_coding_assistant.config.env_loader import EnvLoader

        env_loader = EnvLoader()
        loader = ConfigLoader(env_loader=env_loader)

        assert loader.env_loader is env_loader

    def test_config_loader_default_env_loader(self):
        """Test ConfigLoader creates default EnvLoader when none provided."""
        from local_coding_assistant.config.env_loader import EnvLoader

        loader = ConfigLoader()

        assert loader.env_loader is not None
        assert isinstance(loader.env_loader, EnvLoader)

    def test_app_config_to_dict(self):
        """Test AppConfig conversion to dictionary."""
        config = AppConfig(
            llm=LLMConfig(model_name="gpt-4"),
            runtime=RuntimeConfig(persistent_sessions=True),
        )

        config_dict = config.to_dict()
        assert config_dict["llm"]["model_name"] == "gpt-4"
        assert config_dict["runtime"]["persistent_sessions"] is True


class TestConfigLoader:
    """Test ConfigLoader functionality."""

    def test_config_loader_initialization(self):
        """Test ConfigLoader initialization."""
        loader = ConfigLoader()
        assert loader.config_paths == []
        assert loader._defaults_path.exists()

    def test_config_loader_with_custom_paths(self):
        """Test ConfigLoader with custom config paths."""
        config_paths = [Path("test1.yaml"), Path("test2.yaml")]
        loader = ConfigLoader(config_paths)
        assert loader.config_paths == config_paths

    def test_load_defaults(self):
        """Test loading default configuration."""
        loader = ConfigLoader()

        # Should load from defaults.yaml
        defaults = loader._load_defaults()
        assert "llm" in defaults
        assert "runtime" in defaults
        assert defaults["llm"]["model_name"] == "gpt-5-mini"

    def test_load_defaults_missing_file(self):
        """Test loading defaults when file doesn't exist."""
        with patch.object(Path, "exists", return_value=False):
            loader = ConfigLoader()
            defaults = loader._load_defaults()
            assert defaults == {}

    def test_load_yaml_file_success(self):
        """Test successful YAML file loading."""
        test_data = {"test": "value", "number": 42}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(test_data, f)
            temp_path = Path(f.name)

        try:
            loader = ConfigLoader()
            result = loader._load_yaml_file(temp_path)
            assert result == test_data
        finally:
            temp_path.unlink()

    def test_load_yaml_file_not_found(self):
        """Test YAML file loading when file doesn't exist."""
        loader = ConfigLoader()
        with pytest.raises(ConfigError, match="Config file not found"):
            loader._load_yaml_file(Path("nonexistent.yaml"))

    def test_load_yaml_file_invalid_yaml(self):
        """Test YAML file loading with invalid YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = Path(f.name)
            f.flush()  # Ensure all data is written to disk
            os.fsync(f.fileno())  # Force write to disk

        try:
            loader = ConfigLoader()
            with pytest.raises(ConfigError, match="Invalid YAML"):
                loader._load_yaml_file(temp_path)
        finally:
            # Ensure file is closed and add retry for Windows file handle issues
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    temp_path.unlink()
                    break
                except PermissionError:
                    if attempt < max_retries - 1:
                        time.sleep(0.1)  # Wait a bit for file handles to be released
                    else:
                        # On final attempt, try to force close any remaining handles
                        import gc

                        gc.collect()
                        time.sleep(0.2)
                        temp_path.unlink(missing_ok=True)

    def test_load_env_vars_empty(self):
        """Test loading environment variables when none are set."""
        loader = ConfigLoader()
        env_data = loader.env_loader.get_config_from_env()
        assert env_data == {}

    def test_load_env_vars_locca_prefix(self):
        """Test loading environment variables with LOCCA_ prefix."""
        test_env = {
            "LOCCA_LLM__MODEL_NAME": "gpt-4",
            "LOCCA_RUNTIME__PERSISTENT_SESSIONS": "true",
            "LOCCA_RUNTIME__LOG_LEVEL": "DEBUG",
        }

        with patch.dict(os.environ, test_env):
            loader = ConfigLoader()
            env_data = loader.env_loader.get_config_from_env()

        expected = {
            "llm": {"model_name": "gpt-4"},
            "runtime": {"persistent_sessions": True, "log_level": "DEBUG"},
        }
        assert env_data == expected

    def test_convert_env_value_types(self):
        """Test environment value type conversion."""
        loader = ConfigLoader()

        # Test boolean conversion
        assert loader.env_loader._convert_env_value("true") is True
        assert loader.env_loader._convert_env_value("false") is False

        # Test integer conversion
        assert loader.env_loader._convert_env_value("42") == 42
        assert loader.env_loader._convert_env_value("0") == 0

        # Test float conversion
        assert loader.env_loader._convert_env_value("3.14") == 3.14

        # Test string fallback
        assert loader.env_loader._convert_env_value("hello") == "hello"

    def test_deep_merge_simple(self):
        """Test simple deep merge."""
        loader = ConfigLoader()
        base = {"a": 1, "b": 2}
        overlay = {"b": 3, "c": 4}

        result = loader._deep_merge(base, overlay)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_deep_merge_nested(self):
        """Test nested deep merge."""
        loader = ConfigLoader()
        base = {"llm": {"model": "gpt-3", "temp": 0.7}}
        overlay = {"llm": {"temp": 0.8, "max_tokens": 100}}

        result = loader._deep_merge(base, overlay)
        expected = {"llm": {"model": "gpt-3", "temp": 0.8, "max_tokens": 100}}
        assert result == expected

    def test_load_config_full_integration(self):
        """Test full configuration loading integration."""
        # Create temporary YAML file
        test_config = {
            "llm": {"model_name": "gpt-4"},
            "runtime": {"persistent_sessions": True},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(test_config, f)
            temp_path = Path(f.name)

        try:
            loader = ConfigLoader([temp_path])
            config = loader.load_config()

            # Should merge defaults + YAML + env (no env set)
            assert config.llm.model_name == "gpt-4"  # From YAML
            assert config.runtime.persistent_sessions is True  # From YAML
            assert config.llm.provider == "openai"  # From defaults
        finally:
            temp_path.unlink()

    def test_load_config_with_env_override(self):
        """Test configuration loading with environment variable override."""
        test_env = {
            "LOCCA_LLM__MODEL_NAME": "gpt-4-turbo",
            "LOCCA_RUNTIME__PERSISTENT_SESSIONS": "true",
        }

        with patch.dict(os.environ, test_env):
            loader = ConfigLoader()
            config = loader.load_config()

            # Should use env values over defaults
            assert config.llm.model_name == "gpt-4-turbo"
            assert config.runtime.persistent_sessions is True

    def test_load_config_validation_error(self):
        """Test configuration loading with validation error."""
        # Create invalid config that fails validation
        invalid_config = {
            "llm": {"temperature": -1},  # Invalid temperature
            "runtime": {"max_session_history": 0},  # Invalid max_session_history
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(invalid_config, f)
            temp_path = Path(f.name)

        try:
            loader = ConfigLoader([temp_path])
            with pytest.raises(ConfigError, match="Invalid configuration"):
                loader.load_config()
        finally:
            temp_path.unlink()


class TestGlobalFunctions:
    """Test global configuration functions."""

    def test_get_config_loader_singleton(self):
        """Test that get_config_loader returns the same instance."""
        from local_coding_assistant.config.loader import get_config_loader

        loader1 = get_config_loader()
        loader2 = get_config_loader()
        assert loader1 is loader2

    def test_load_config_function(self):
        """Test the global load_config function."""
        from local_coding_assistant.config.loader import load_config

        config = load_config()
        assert isinstance(config, AppConfig)
        assert config.llm.model_name == "gpt-5-mini"
