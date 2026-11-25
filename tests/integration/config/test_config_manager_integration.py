"""Integration tests for the ConfigManager system."""

from local_coding_assistant.agent.llm.service import LLMService
from local_coding_assistant.config.config_manager import ConfigManager
from local_coding_assistant.config.field import config_field_registry
from local_coding_assistant.config.schemas import LLMConfig


class TestConfigManagerIntegration:
    """Integration tests for ConfigManager with other components."""

    def test_llm_manager_integration(self):
        """Test that LLMService works with ConfigManager."""
        # Save and restore the registry so this test does not leak state into
        # subsequently executed test modules (the global ConfigFieldRegistry
        # is shared across the whole session).
        saved_fields = dict(config_field_registry._fields)
        config_field_registry.clear()
        try:
            manager = ConfigManager()
            manager.load_global_config()

            llm_service = LLMService(manager)

            # Should be able to get LLM config through config_manager
            resolved_config = llm_service._config_manager.global_config
            llm_config = resolved_config.llm
            assert isinstance(llm_config, LLMConfig)
            # The model_name is not a direct field in LLMConfig, it's resolved through providers
            # The default model would come from the provider system or agent policies
            assert isinstance(llm_config, LLMConfig)
        finally:
            config_field_registry._fields.clear()
            config_field_registry._fields.update(saved_fields)
        # The model_name is not a direct field in LLMConfig, it's resolved through providers
        # The default model would come from the provider system or agent policies

        assert isinstance(llm_config, LLMConfig)
