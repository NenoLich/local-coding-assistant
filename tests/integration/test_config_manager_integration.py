"""Integration tests for the ConfigManager system."""

from local_coding_assistant.agent import LLMService
from local_coding_assistant.config import ConfigManager
from local_coding_assistant.config.schemas import LLMConfig


class TestConfigManagerIntegration:
    """Integration tests for ConfigManager with other components."""

    def test_llm_manager_integration(self):
        """Test that LLMService works with ConfigManager."""
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
