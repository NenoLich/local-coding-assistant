"""Unit tests for the ConfigBuilder class."""

from unittest.mock import Mock, patch

import pytest

from local_coding_assistant.config.builder import ConfigBuilder
from local_coding_assistant.config.schemas import AppConfig, ToolConfigList
from local_coding_assistant.core.exceptions import ConfigError


class TestConfigBuilder:
    """Test cases for ConfigBuilder functionality."""

    @pytest.fixture
    def base_config(self):
        """Create a basic AppConfig for testing."""
        return AppConfig()

    @pytest.fixture
    def validation_engine(self):
        """Create a mock ValidationEngine."""
        mock_engine = Mock()
        # Return the input value as validated by default
        mock_engine.get_validated_value.side_effect = (
            lambda path, value, **kwargs: value
        )
        return mock_engine

    @pytest.fixture
    def config_builder(self, base_config, validation_engine):
        """Create a ConfigBuilder instance."""
        return ConfigBuilder(base_config, validation_engine)

    def test_init(self, base_config, validation_engine):
        """Test ConfigBuilder initialization."""
        builder = ConfigBuilder(base_config, validation_engine)

        assert builder._base_config == base_config
        assert builder._validation_engine == validation_engine
        assert builder._allow_deferring is True
        assert builder._session_overrides == {}
        assert builder._call_overrides == {}
        assert builder._built_config is None
        assert builder._is_base_config_validated is False

    def test_init_with_allow_deferring_false(self, base_config, validation_engine):
        """Test ConfigBuilder initialization with allow_deferring=False."""
        builder = ConfigBuilder(base_config, validation_engine, allow_deferring=False)

        assert builder._allow_deferring is False

    def test_get_field_value(self, config_builder):
        """Test _get_field_value method."""
        config = AppConfig()
        config.llm.temperature = 0.8

        result = config_builder._get_field_value(config, "llm.temperature")
        assert result == 0.8

    def test_get_field_value_nested(self, config_builder):
        """Test _get_field_value with deeper nesting."""
        config = AppConfig()
        config.runtime.enable_logging = True

        result = config_builder._get_field_value(config, "runtime.enable_logging")
        assert result is True

    def test_set_nested_value(self, config_builder):
        """Test _set_nested_value method."""
        data = {"llm": {"temperature": 0.7}}
        config_builder._set_nested_value(data, "llm.temperature", 0.9)
        assert data["llm"]["temperature"] == 0.9

    def test_set_nested_value_create_path(self, config_builder):
        """Test _set_nested_value creates nested dicts."""
        data = {}
        config_builder._set_nested_value(data, "llm.temperature", 0.8)
        assert data["llm"]["temperature"] == 0.8

    def test_set_session_overrides(self, config_builder, validation_engine):
        """Test setting session overrides."""
        validation_engine.get_validated_value.side_effect = (
            lambda path, value, **kwargs: value
        )

        overrides = {"llm.temperature": 0.5}
        config_builder.set_session_overrides(overrides)

        assert config_builder._session_overrides == overrides
        assert config_builder._built_config is None  # Cache invalidated

    def test_set_session_overrides_skip_validation(self, config_builder):
        """Test setting session overrides with skip_validation=True."""
        overrides = {"llm.temperature": 0.5}
        config_builder.set_session_overrides(overrides, skip_validation=True)

        assert config_builder._session_overrides == overrides

    def test_set_session_overrides_invalid_value(
        self, config_builder, validation_engine
    ):
        """Test setting session overrides with invalid value."""
        validation_engine.get_validated_value.return_value = None
        validation_engine.get_validated_value.side_effect = (
            lambda path, value, **kwargs: None
        )

        overrides = {"llm.temperature": "invalid"}
        config_builder.set_session_overrides(overrides)

        assert config_builder._session_overrides == {}  # No valid overrides

    def test_set_call_overrides(self, config_builder, validation_engine):
        """Test setting call overrides."""
        validation_engine.get_validated_value.side_effect = (
            lambda path, value, **kwargs: value
        )

        overrides = {"llm.temperature": 0.6}
        result = config_builder.set_call_overrides(overrides)

        assert isinstance(result, AppConfig)
        assert config_builder._call_overrides == {}  # Cleared after build
        assert config_builder._built_config is None  # Cache invalidated

    def test_set_call_overrides_invalid_value(self, config_builder, validation_engine):
        """Test setting call overrides with invalid value."""
        validation_engine.get_validated_value.side_effect = (
            lambda path, value, **kwargs: None if path == "llm.temperature" else value
        )

        overrides = {"llm.temperature": "invalid"}
        result = config_builder.set_call_overrides(overrides)

        assert isinstance(result, AppConfig)
        assert config_builder._call_overrides == {}  # Cleared after build

    def test_clear_session_overrides(self, config_builder):
        """Test clearing session overrides."""
        config_builder._session_overrides = {"llm.temperature": 0.5}
        config_builder._built_config = AppConfig()  # Mock built config

        config_builder.clear_session_overrides()

        assert config_builder._session_overrides == {}
        assert config_builder._built_config is None

    def test_clear_call_overrides(self, config_builder):
        """Test clearing call overrides."""
        config_builder._call_overrides = {"llm.temperature": 0.5}
        config_builder._built_config = AppConfig()  # Mock built config

        config_builder.clear_call_overrides()

        assert config_builder._call_overrides == {}
        assert config_builder._built_config is None

    def test_get_session_overrides(self, config_builder):
        """Test getting session overrides."""
        overrides = {"llm.temperature": 0.5}
        config_builder._session_overrides = overrides

        result = config_builder.get_session_overrides()

        assert result == overrides
        assert result is not config_builder._session_overrides  # Should be a copy

    def test_build_caches_result(self, config_builder):
        """Test that build caches the result."""
        result1 = config_builder.build()
        result2 = config_builder.build()

        assert result1 is result2  # Same object returned

    def test_build_invalidates_cache_on_overrides(
        self, config_builder, validation_engine
    ):
        """Test that setting overrides invalidates cache."""
        config_builder.build()  # Cache the config
        assert config_builder._built_config is not None

        validation_engine.get_validated_value.side_effect = (
            lambda path, value, **kwargs: value
        )
        config_builder.set_session_overrides({"llm.temperature": 0.5})

        assert config_builder._built_config is None

    def test_update_base_config_with_config_model(self, config_builder):
        """Test updating base config with another ConfigModel."""
        new_llm_config = AppConfig().llm
        new_llm_config.temperature = 0.9

        config_builder.update_base_config(new_llm_config, "llm")

        assert config_builder._base_config.llm.temperature == 0.9
        assert config_builder._built_config is None  # Cache invalidated
        assert config_builder._is_base_config_validated is False

    def test_update_base_config_with_dict(self, config_builder):
        """Test updating base config with dict."""
        # This sets the llm field to the dict
        config_builder.update_base_config({"temperature": 0.8}, "llm")

        assert config_builder._base_config.llm == {"temperature": 0.8}

    def test_update_base_config_with_dict_no_field_path(self, config_builder):
        """Test updating base config with dict without field_path."""
        # This replaces the llm field with the dict
        config_builder.update_base_config({"llm": {"temperature": 0.7}})

        # The llm field should be replaced with the dict
        assert config_builder._base_config.llm == {"temperature": 0.7}

    def test_update_base_config_direct_assignment(self, config_builder):
        """Test updating base config with direct assignment."""
        config_builder.update_base_config(0.6, "llm.temperature")

        assert config_builder._base_config.llm.temperature == 0.6

    def test_update_tools(self, config_builder):
        """Test updating tools configuration."""
        tools = {
            "tool1": {
                "id": "test_tool",
                "description": "A test tool",
                "name": "test_tool",
                "enabled": True,
            }
        }
        config_builder.update_tools(tools)

        assert isinstance(config_builder._base_config.tools, ToolConfigList)
        assert len(config_builder._base_config.tools.tools) == 1
        assert config_builder._base_config.tools.tools[0].id == "test_tool"
        assert config_builder._built_config is None
        assert config_builder._is_base_config_validated is False

    def test_update_tools_no_tools_field(self, config_builder):
        """Test updating tools when base config has no tools field."""
        # Mock base config without tools field
        config_builder._base_config = Mock(spec=AppConfig)
        del config_builder._base_config.tools

        with pytest.raises(
            ConfigError, match="Base config does not have a tools field"
        ):
            config_builder.update_tools({})

    def test_validate_base_config_no_fields_with_deps(self, config_builder):
        """Test base config validation when no fields have dependencies."""
        with patch(
            "local_coding_assistant.config.builder.config_field_registry"
        ) as mock_registry:
            mock_registry.get_fields_with_dependencies.return_value = []

            config_builder._validate_base_config()

            assert config_builder._is_base_config_validated is True

    def test_validate_base_config_with_validation_errors(
        self, config_builder, validation_engine
    ):
        """Test base config validation with validation errors."""
        with patch(
            "local_coding_assistant.config.builder.config_field_registry"
        ) as mock_registry:
            mock_field = Mock()
            mock_field.full_path = "test.field"
            mock_registry.get_fields_with_dependencies.return_value = [mock_field]

            validation_engine.get_validated_value.return_value = None

            with pytest.raises(ConfigError, match="Base config validation failed"):
                config_builder._validate_base_config()

    @patch("local_coding_assistant.config.builder.logger")
    def test_validate_base_config_with_fallback(
        self, mock_logger, config_builder, validation_engine
    ):
        """Test base config validation adds fallback to session overrides."""
        with patch(
            "local_coding_assistant.config.builder.config_field_registry"
        ) as mock_registry:
            mock_field = Mock()
            mock_field.full_path = "test.field"
            mock_registry.get_fields_with_dependencies.return_value = [mock_field]

            config_builder._get_field_value = Mock(return_value="original")
            validation_engine.get_validated_value.side_effect = (
                lambda path, value, **kwargs: "fallback"
                if path == "test.field"
                else value
            )

            config_builder._validate_base_config()

            assert config_builder._session_overrides["test.field"] == "fallback"
            mock_logger.info.assert_called()

    def test_build_calls_validate_base_config(self, config_builder):
        """Test that build calls _validate_base_config when not validated."""
        with patch.object(config_builder, "_validate_base_config") as mock_validate:
            config_builder._is_base_config_validated = False
            config_builder.build()

            mock_validate.assert_called_once()

    def test_build_does_not_call_validate_when_already_validated(self, config_builder):
        """Test that build does not re-validate when already validated."""
        with patch.object(config_builder, "_validate_base_config") as mock_validate:
            config_builder._is_base_config_validated = True
            config_builder.build()

            mock_validate.assert_not_called()
