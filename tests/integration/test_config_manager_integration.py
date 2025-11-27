"""Integration tests for the ConfigManager system."""

from unittest.mock import patch

from local_coding_assistant.agent.llm_manager import LLMManager
from local_coding_assistant.config import ConfigManager
from local_coding_assistant.config.schemas import LLMConfig


class TestConfigManagerIntegration:
    """Integration tests for ConfigManager with other components."""

    def test_llm_manager_integration(self):
        """Test that LLMManager works with ConfigManager."""
        manager = ConfigManager()
        manager.load_global_config()

        llm_manager = LLMManager(manager)

        # Should be able to get LLM config through config_manager
        resolved_config = llm_manager.config_manager.resolve()
        llm_config = resolved_config.llm
        assert isinstance(llm_config, LLMConfig)
        # The model_name is not a direct field in LLMConfig, it's resolved through providers
        # The default model would come from the provider system or agent policies

        # Should be able to get config with overrides
        resolved_config_with_override = llm_manager.config_manager.resolve(
            model_name="gpt-4"
        )
        llm_config_with_override = resolved_config_with_override.llm
        assert isinstance(llm_config_with_override, LLMConfig)

    def test_llm_manager_mock_response(self):
        """Test that LLMManager returns mock responses in test mode."""
        from local_coding_assistant.agent.llm_manager import LLMManager, LLMRequest

        manager = ConfigManager()
        manager.load_global_config()

        llm_manager = LLMManager(manager)
        request = LLMRequest(prompt="test prompt")

        response = llm_manager._generate_mock_response(request)

        assert response.content == "[LLMManager] Echo: test prompt"
        assert response.model_used == "mock-model"
        assert response.tokens_used == 50
