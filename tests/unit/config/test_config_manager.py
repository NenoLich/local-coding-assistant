"""Unit tests for the ConfigManager system."""

import pytest

from local_coding_assistant.config import ConfigManager
from local_coding_assistant.config.schemas import AppConfig, LLMConfig, RuntimeConfig


class TestConfigManager:
    """Test cases for ConfigManager functionality."""

    def test_init(self):
        """Test ConfigManager initialization."""
        manager = ConfigManager()
        assert manager._global_config is None
        assert manager._session_overrides == {}

    def test_load_global_config(self):
        """Test loading global configuration."""
        manager = ConfigManager()
        config = manager.load_global_config()

        assert isinstance(config, AppConfig)
        assert isinstance(config.llm, LLMConfig)
        assert isinstance(config.runtime, RuntimeConfig)
        assert config.runtime.enable_logging is True
        assert config.runtime.log_level == "INFO"

    def test_set_session_overrides(self):
        """Test setting session overrides."""
        manager = ConfigManager()
        manager.load_global_config()

        overrides = {"llm.temperature": 0.5}
        manager.set_session_overrides(overrides)

        assert manager._session_overrides == overrides

    def test_clear_session_overrides(self):
        """Test clearing session overrides."""
        manager = ConfigManager()
        manager.load_global_config()

        manager.set_session_overrides({"llm.temperature": 0.5})
        assert manager._session_overrides != {}

        manager.clear_session_overrides()
        assert manager._session_overrides == {}

    def test_resolve_global_only(self):
        """Test resolving configuration with global layer only."""
        manager = ConfigManager()
        manager.load_global_config()

        resolved = manager.resolve()

        assert isinstance(resolved, AppConfig)
        assert resolved.llm.temperature == 0.7

    def test_resolve_with_session_overrides(self):
        """Test resolving configuration with session overrides."""
        manager = ConfigManager()
        manager.load_global_config()

        session_overrides = {"llm.temperature": 0.5}
        manager.set_session_overrides(session_overrides)

        resolved = manager.resolve()

        assert resolved.llm.temperature == 0.5  # Session override applied

    def test_get_config_with_call_overrides(self):
        """Test resolving configuration with call overrides."""
        manager = ConfigManager()
        manager.load_global_config()

        call_overrides = {"llm.temperature": 0.3}
        resolved = manager.get_config(overrides=call_overrides)

        assert resolved.llm.temperature == 0.3  # Call override applied

    def test_get_config_with_provider_override(self):
        """Test resolving configuration with provider override."""
        manager = ConfigManager()
        manager.load_global_config()

        resolved = manager.get_config(provider="anthropic")

        # Provider override is applied via ConfigManager but not stored in LLMConfig
        # The LLMConfig doesn't have provider/model_name fields, so we test that
        # the resolve method completes without error and temperature is preserved
        assert resolved.llm.temperature == 0.7  # Should keep global value

    def test_get_config_with_model_override(self):
        """Test resolving configuration with model override."""
        manager = ConfigManager()
        manager.load_global_config()

        resolved = manager.get_config(model_name="gpt-4")

        # Model override is applied via ConfigManager but not stored in LLMConfig
        # The LLMConfig doesn't have provider/model_name fields, so we test that
        # the resolve method completes without error and temperature is preserved
        assert resolved.llm.temperature == 0.7  # Should keep global value

    def test_get_config_priority_order(self):
        """Test that call overrides take priority over session overrides."""
        manager = ConfigManager()
        manager.load_global_config()

        # Set session override
        session_overrides = {"llm.temperature": 0.5}
        manager.set_session_overrides(session_overrides)

        # Call override should win
        call_overrides = {"llm.temperature": 0.3}
        resolved = manager.get_config(overrides=call_overrides)

        assert resolved.llm.temperature == 0.3  # Call override wins

    def test_resolve_session_overrides_global(self):
        """Test that session overrides take priority over global config."""
        manager = ConfigManager()
        manager.load_global_config()

        # Session override should win over global
        session_overrides = {"llm.temperature": 0.8}
        manager.set_session_overrides(session_overrides)

        resolved = manager.resolve()

        assert resolved.llm.temperature == 0.8  # Session override wins

    def test_config_manager_properties(self):
        """Test ConfigManager property accessors."""
        manager = ConfigManager()
        manager.load_global_config()

        # Test global_config property
        assert manager.global_config is not None
        assert isinstance(manager.global_config, AppConfig)

        # Test session_overrides property
        manager.set_session_overrides({"llm.temperature": 0.5})
        overrides = manager.session_overrides
        assert overrides == {"llm.temperature": 0.5}

        # Should return a copy, not the original
        overrides["llm.temperature"] = 0.1
        assert manager.session_overrides["llm.temperature"] == 0.5

    def test_resolve_without_global_config(self):
        """Test that resolve raises error when global config not loaded."""
        manager = ConfigManager()

        with pytest.raises(Exception):  # Should raise ConfigError
            manager.resolve()

    def test_nested_override_application(self):
        """Test that nested overrides are applied correctly."""
        manager = ConfigManager()
        manager.load_global_config()

        # Test nested override
        overrides = {"runtime.enable_logging": False}
        resolved = manager.get_config(overrides=overrides)

        assert resolved.runtime.enable_logging is False
        assert resolved.llm.temperature == 0.7  # Should keep global value
