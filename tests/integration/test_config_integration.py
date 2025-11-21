"""Integration tests for configuration management."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from local_coding_assistant.agent import LLMManager
from local_coding_assistant.core.exceptions import AgentError
from local_coding_assistant.runtime import RuntimeManager


class TestBootstrapIntegration:
    """Integration tests for bootstrap functionality."""

    def test_bootstrap_uses_default_config(self, ctx):
        """Test that bootstrap uses default configuration."""
        # The ctx fixture calls bootstrap() which loads config
        # Check that the runtime manager was created with config
        runtime = ctx.get("runtime")
        assert runtime is not None
        assert hasattr(runtime, "config_manager")
        # Test that config_manager can resolve runtime config
        resolved_config = runtime.config_manager.resolve()
        assert resolved_config.runtime.persistent_sessions is False  # Default value

    def test_bootstrap_with_custom_config_file(self):
        """Test bootstrap with a custom configuration file."""
        # Create temporary config file with new provider-based structure
        custom_config = {
            "llm": {"temperature": 0.8, "max_tokens": 2000, "max_retries": 5},
            "runtime": {"persistent_sessions": True, "log_level": "DEBUG"},
            "providers": {
                "openai": {
                    "name": "openai",
                    "driver": "openai_chat",
                    "base_url": "https://api.openai.com/v1/chat/completions",
                    "api_key_env": "OPENAI_API_KEY",
                    "health_check_endpoint": "https://api.openai.com/v1/models",
                    "models": {
                        "gpt-4.1": {
                            "supported_parameters": [
                                "max_tokens",
                                "temperature",
                                "top_p",
                            ]
                        }
                    },
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(custom_config, f)
            config_path = Path(f.name)

        try:
            from local_coding_assistant.core.bootstrap import bootstrap

            # Bootstrap with custom config
            ctx = bootstrap(config_path=str(config_path))

            # Check that custom config was applied
            llm = ctx.get("llm")
            runtime = ctx.get("runtime")

            assert llm is not None
            assert runtime is not None

            # Check LLM config via config manager
            llm_resolved = llm.config_manager.resolve()
            assert llm_resolved.llm.temperature == 0.8
            assert llm_resolved.llm.max_tokens == 2000
            assert llm_resolved.llm.max_retries == 5
            assert llm_resolved.runtime.persistent_sessions is True
            assert llm_resolved.runtime.log_level == "DEBUG"

            # Check provider configuration
            assert len(llm_resolved.providers) == 1
            openai_provider = llm_resolved.providers["openai"]
            assert openai_provider.name == "openai"
            assert openai_provider.driver == "openai_chat"
            assert (
                openai_provider.base_url == "https://api.openai.com/v1/chat/completions"
            )
            assert openai_provider.api_key_env == "OPENAI_API_KEY"
            assert (
                openai_provider.health_check_endpoint
                == "https://api.openai.com/v1/models"
            )

            # Check models
            assert len(openai_provider.models) == 1
            model_config = openai_provider.models[0]
            assert model_config.name == "gpt-4.1"
            assert set(model_config.supported_parameters) == {
                "max_tokens",
                "temperature",
                "top_p",
            }

        finally:
            # Clean up temporary file
            if config_path.exists():
                config_path.unlink()

    def test_bootstrap_with_environment_variables(self):
        """Test bootstrap with environment variable configuration."""
        test_env = {
            "LOCCA_LLM__TEMPERATURE": "0.9",
            "LOCCA_LLM__MAX_TOKENS": "3000",
            "LOCCA_LLM__MAX_RETRIES": "7",
            "LOCCA_RUNTIME__PERSISTENT_SESSIONS": "true",
            "LOCCA_RUNTIME__MAX_SESSION_HISTORY": "150",
            "LOCCA_RUNTIME__LOG_LEVEL": "ERROR",
        }

        with patch.dict(os.environ, test_env):
            from local_coding_assistant.core.bootstrap import bootstrap

            ctx = bootstrap()

            # Check that env vars were applied
            llm = ctx.get("llm")
            runtime = ctx.get("runtime")

            assert llm is not None
            assert runtime is not None
            # Check LLM config via config manager
            llm_resolved = llm.config_manager.resolve()
            assert llm_resolved.llm.temperature == 0.9
            assert llm_resolved.llm.max_tokens == 3000
            assert llm_resolved.llm.max_retries == 7

            # Check runtime config via config manager
            runtime_resolved = runtime.config_manager.resolve()
            assert runtime_resolved.runtime.persistent_sessions is True
            assert runtime_resolved.runtime.max_session_history == 150
            assert runtime_resolved.runtime.log_level == "ERROR"

    def test_bootstrap_config_priority_order(self):
        """Test that configuration priority order is correct: ENV > YAML > Defaults."""
        # Create YAML config
        yaml_config = {
            "llm": {"temperature": 0.5, "max_tokens": 1000},
            "runtime": {"persistent_sessions": False, "log_level": "INFO"},
        }

        # Set environment variables (should override YAML)
        env_config = {
            "LOCCA_LLM__TEMPERATURE": "0.9",
            "LOCCA_LLM__MAX_TOKENS": "2000",
            "LOCCA_RUNTIME__PERSISTENT_SESSIONS": "true",
            "LOCCA_RUNTIME__LOG_LEVEL": "DEBUG",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_config, f)
            config_path = Path(f.name)

        try:
            with patch.dict(os.environ, env_config):
                from local_coding_assistant.core.bootstrap import bootstrap

                ctx = bootstrap(config_path=str(config_path))

                # Should use ENV values (highest priority)
                llm = ctx.get("llm")
                runtime = ctx.get("runtime")

                assert llm is not None
                assert runtime is not None
                # Check LLM config via config manager
                llm_resolved = llm.config_manager.resolve()
                assert llm_resolved.llm.temperature == 0.9  # From ENV
                assert llm_resolved.llm.max_tokens == 2000  # From ENV

                # Check runtime config via config manager
                runtime_resolved = runtime.config_manager.resolve()
                assert runtime_resolved.runtime.persistent_sessions is True  # From ENV
                assert runtime_resolved.runtime.log_level == "DEBUG"  # From ENV

        finally:
            config_path.unlink()

    def test_bootstrap_invalid_config_file(self):
        """Test bootstrap with invalid configuration file."""
        # Create invalid YAML file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = Path(f.name)

        try:
            from local_coding_assistant.core.bootstrap import bootstrap
            from local_coding_assistant.core.exceptions import ConfigError

            # Should raise RuntimeError with ConfigError as cause for invalid YAML
            with pytest.raises(RuntimeError) as excinfo:
                bootstrap(config_path=str(config_path))
                
            # The original error should be a ConfigError
            assert isinstance(excinfo.value.__cause__, ConfigError), \
                f"Expected ConfigError, got {type(excinfo.value.__cause__).__name__}"
                
            # The error message should indicate invalid YAML
            error_msg = str(excinfo.value.__cause__).lower()
            assert "invalid yaml" in error_msg or "yaml" in error_msg, \
                f"Expected YAML error in message, got: {error_msg}"

        finally:
            if os.path.exists(config_path):
                config_path.unlink()

    def test_bootstrap_missing_config_file(self):
        """Test bootstrap with missing configuration file."""
        from local_coding_assistant.core.bootstrap import bootstrap
        
        # Should not raise an error for missing config file
        # It should use default configuration
        ctx = bootstrap(config_path="nonexistent.yaml")
        
        # Should still get a valid context with default configuration
        assert ctx is not None
        assert hasattr(ctx, 'get')
        
        # Should have LLM manager initialized with default config
        llm = ctx.get("llm")
        assert llm is not None
        assert hasattr(llm, "config_manager")

    def test_configuration_injection_into_services(self, ctx):
        """Test that configuration is properly injected into services."""
        # Check LLM manager got config
        llm = ctx.get("llm")
        assert isinstance(llm, LLMManager)
        # Check that LLM has config manager
        assert hasattr(llm, "config_manager")

        # Check runtime manager got config
        runtime = ctx.get("runtime")
        assert isinstance(runtime, RuntimeManager)
        # Check that runtime has config manager
        assert hasattr(runtime, "config_manager")

        # Check tool manager is available
        tools = ctx.get("tools")
        assert tools is not None

    def test_configuration_schema_validation(self):
        """Test that configuration schema validation works in integration."""
        # Test with invalid LLM config that should fail validation
        invalid_config = {
            "llm": {
                "temperature": -1,  # Invalid temperature (should be >= 0.0)
                "max_tokens": 0,  # Invalid max_tokens (should be > 0)
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(invalid_config, f)
            config_path = Path(f.name)

        try:
            from local_coding_assistant.core.bootstrap import bootstrap
            from local_coding_assistant.core.exceptions import ConfigError

            # Bootstrap should raise RuntimeError with ConfigError as cause due to invalid configuration
            with pytest.raises(RuntimeError) as excinfo:
                bootstrap(config_path=str(config_path))
            
            # The original error should be a ConfigError
            assert isinstance(excinfo.value.__cause__, ConfigError), \
                f"Expected ConfigError, got {type(excinfo.value.__cause__).__name__}"
            
            # Get the error message from the cause
            error_msg = str(excinfo.value.__cause__)
            
            # Verify the error message contains the validation errors
            assert "Invalid global configuration" in error_msg, \
                f"Expected 'Invalid global configuration' in error message, got: {error_msg}"
            assert "temperature" in error_msg, \
                f"Expected 'temperature' in error message, got: {error_msg}"
            assert "max_tokens" in error_msg, \
                f"Expected 'max_tokens' in error message, got: {error_msg}"
            
        finally:
            # Clean up the temporary file
            if os.path.exists(config_path):
                os.unlink(config_path)

    def test_logging_configuration_application(self):
        """Test that logging configuration is properly applied."""
        # Test with logging disabled
        config = {"runtime": {"enable_logging": False}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            config_path = Path(f.name)

        try:
            from local_coding_assistant.core.bootstrap import bootstrap

            # Bootstrap should work even with logging disabled
            ctx = bootstrap(config_path=str(config_path))

            # Runtime should have logging disabled
            runtime = ctx.get("runtime")
            assert runtime is not None
            # Check runtime config via config manager
            runtime_resolved = runtime.config_manager.resolve()
            assert runtime_resolved.runtime.enable_logging is False

        finally:
            config_path.unlink()


class TestConfigurationEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_yaml_file(self):
        """Test loading empty YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")  # Empty file
            config_path = Path(f.name)

        try:
            from local_coding_assistant.core.bootstrap import bootstrap

            # Should fall back to defaults
            ctx = bootstrap(config_path=str(config_path))

            # Should use default values
            llm = ctx.get("llm")
            assert llm is not None
            # Check LLM config via config manager
            llm_resolved = llm.config_manager.resolve()
            assert llm_resolved.llm.temperature == 0.7  # Default value
            assert llm_resolved.llm.max_tokens == 1000  # Default value
            assert llm_resolved.llm.max_retries == 3  # Default value

        finally:
            config_path.unlink()

    def test_yaml_with_null_values(self):
        """Test YAML file with null values."""
        config = {"llm": {"max_tokens": None}, "runtime": {"log_level": None}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            config_path = Path(f.name)

        try:
            from local_coding_assistant.core.bootstrap import bootstrap

            ctx = bootstrap(config_path=str(config_path))

            # Should handle null values properly
            llm = ctx.get("llm")
            assert llm is not None
            # Check LLM config via config manager
            llm_resolved = llm.config_manager.resolve()
            assert llm_resolved.llm.max_tokens is None

        finally:
            config_path.unlink()

    def test_environment_variable_type_conversion(self):
        """Test that environment variables are properly converted to correct types."""
        test_env = {
            "LOCCA_LLM__TEMPERATURE": "0.8",
            "LOCCA_LLM__MAX_TOKENS": "1500",
            "LOCCA_RUNTIME__MAX_SESSION_HISTORY": "200",
            "LOCCA_RUNTIME__PERSISTENT_SESSIONS": "true",
        }

        with patch.dict(os.environ, test_env):
            from local_coding_assistant.core.bootstrap import bootstrap

            ctx = bootstrap()

            # Check types are correct
            llm = ctx.get("llm")
            runtime = ctx.get("runtime")

            assert llm is not None
            # Check LLM config via config manager
            llm_resolved = llm.config_manager.resolve()
            assert isinstance(llm_resolved.llm.temperature, float)
            assert llm_resolved.llm.temperature == 0.8
            assert isinstance(llm_resolved.llm.max_tokens, int)
            assert llm_resolved.llm.max_tokens == 1500

            assert runtime is not None
            # Check runtime config via config manager
            runtime_resolved = runtime.config_manager.resolve()
            assert isinstance(runtime_resolved.runtime.max_session_history, int)
            assert runtime_resolved.runtime.max_session_history == 200

    @pytest.mark.asyncio
    async def test_runtime_manager_orchestrate_with_model_override(
        self, ctx_with_mocked_llm
    ):
        """Test RuntimeManager orchestrate method with model override."""
        runtime = ctx_with_mocked_llm.get("runtime")

        # Test with model override - using correct API without provider parameter
        result = await runtime.orchestrate("test query", model="gpt-4.1")

        # Should complete successfully
        assert "message" in result
        assert "model_used" in result
        assert result["model_used"] == "gpt-4.1"

    @pytest.mark.asyncio
    async def test_runtime_manager_orchestrate_with_multiple_overrides(
        self, ctx_with_mocked_llm
    ):
        """Test RuntimeManager orchestrate method with multiple configuration overrides."""
        runtime = ctx_with_mocked_llm.get("runtime")

        # Test with model override
        result_first_run = await runtime.orchestrate("test query", model="gpt-4.1")

        # Should complete successfully
        assert "message" in result_first_run
        assert "model_used" in result_first_run
        assert result_first_run["model_used"] == "gpt-4.1"

        result_second_run = await runtime.orchestrate("test query", model="gpt-5-mini")

        assert "model_used" in result_second_run
        assert result_second_run["model_used"] == "gpt-5-mini"

    @pytest.mark.asyncio
    async def test_runtime_manager_orchestrate_config_validation(
        self, ctx_with_mocked_llm
    ):
        """Test RuntimeManager orchestrate method with invalid configuration."""
        runtime = ctx_with_mocked_llm.get("runtime")

        # Test invalid temperature
        with pytest.raises(AgentError, match="Configuration update validation failed"):
            await runtime.orchestrate("test query", temperature=-1)

        # Test invalid max_tokens
        with pytest.raises(AgentError, match="Configuration update validation failed"):
            await runtime.orchestrate("test query", max_tokens=0)

    @pytest.mark.asyncio
    async def test_runtime_manager_config_persistence_across_calls(
        self, ctx_with_mocked_llm
    ):
        """Test that configuration overrides don't affect the base configuration."""
        runtime = ctx_with_mocked_llm.get("runtime")
        # Get initial LLM config via config manager
        initial_llm_config = runtime._llm_manager.config_manager.resolve()
        initial_temperature = initial_llm_config.llm.temperature
        initial_max_tokens = initial_llm_config.llm.max_tokens

        # Call with override
        await runtime.orchestrate(
            "test query", model="gpt-5-mini", temperature=0.9, max_tokens=1000
        )

        # Base config should remain unchanged
        updated_llm_config = runtime._llm_manager.config_manager.resolve()
        assert updated_llm_config.llm.temperature == initial_temperature
        assert updated_llm_config.llm.max_tokens == initial_max_tokens

    @pytest.mark.asyncio
    async def test_runtime_manager_handles_quota_errors_gracefully(
        self, ctx_with_mocked_llm
    ):
        """Test that RuntimeManager handles quota exceeded errors gracefully."""
        runtime = ctx_with_mocked_llm.get("runtime")
        # This test uses mocked LLM manager to simulate quota errors

        # Mock the LLM manager to raise a quota error
        from unittest.mock import AsyncMock

        from local_coding_assistant.core.exceptions import AgentError

        async def mock_generate_quota_error(*args, **kwargs):
            # Simulate a quota error that should be caught by the test
            raise AgentError(
                "insufficient_quota: Your OpenAI API quota has been exceeded"
            )

        runtime._llm_manager.generate = AsyncMock(side_effect=mock_generate_quota_error)

        # The test should catch the AgentError and handle it appropriately
        # Since we're mocking the quota error, the test should pass
        with pytest.raises(AgentError) as exc_info:
            await runtime.orchestrate("test query", model="gpt-4.1")

        # Verify it's a quota-related error
        assert "insufficient_quota" in str(exc_info.value)
