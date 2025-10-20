"""Integration tests for configuration management."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from local_coding_assistant.agent import LLMManager
from local_coding_assistant.core.exceptions import AgentError, ConfigError
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
        # Create temporary config file
        custom_config = {
            "llm": {"model_name": "gpt-4.1", "temperature": 0.8},
            "runtime": {"persistent_sessions": True, "log_level": "DEBUG"},
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
            assert llm_resolved.llm.model_name == "gpt-4.1"
            assert llm_resolved.llm.temperature == 0.8
            # Check runtime config via config manager
            runtime_resolved = runtime.config_manager.resolve()
            assert runtime_resolved.runtime.persistent_sessions is True
            assert runtime_resolved.runtime.log_level == "DEBUG"

        finally:
            config_path.unlink()

    def test_bootstrap_with_environment_variables(self):
        """Test bootstrap with environment variable configuration."""
        test_env = {
            "LOCCA_LLM__MODEL_NAME": "gpt-4-turbo",
            "LOCCA_RUNTIME__PERSISTENT_SESSIONS": "true",
            "LOCCA_RUNTIME__MAX_SESSION_HISTORY": "50",
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
            assert llm_resolved.llm.model_name == "gpt-4-turbo"
            # Check runtime config via config manager
            runtime_resolved = runtime.config_manager.resolve()
            assert runtime_resolved.runtime.persistent_sessions is True
            assert runtime_resolved.runtime.max_session_history == 50

    def test_bootstrap_config_priority_order(self):
        """Test that configuration priority order is correct: ENV > YAML > Defaults."""
        # Create YAML config
        yaml_config = {
            "llm": {"model_name": "gpt-4-from-yaml"},
            "runtime": {"persistent_sessions": False},
        }

        # Set environment variables (should override YAML)
        env_config = {
            "LOCCA_LLM__MODEL_NAME": "gpt-4-from-env",
            "LOCCA_RUNTIME__PERSISTENT_SESSIONS": "true",
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
                assert llm_resolved.llm.model_name == "gpt-4-from-env"  # From ENV
                # Check runtime config via config manager
                runtime_resolved = runtime.config_manager.resolve()
                assert runtime_resolved.runtime.persistent_sessions is True  # From ENV

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

            # Should raise ConfigError for invalid YAML
            with pytest.raises(ConfigError):
                bootstrap(config_path=str(config_path))

        finally:
            config_path.unlink()

    def test_bootstrap_missing_config_file(self):
        """Test bootstrap with missing configuration file."""
        from local_coding_assistant.core.bootstrap import bootstrap

        # Should raise ConfigError for missing file
        with pytest.raises(ConfigError):
            bootstrap(config_path="nonexistent.yaml")

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
                "temperature": -1,  # Invalid temperature
                "max_tokens": 0,  # Invalid max_tokens
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(invalid_config, f)
            config_path = Path(f.name)

        try:
            from local_coding_assistant.core.bootstrap import bootstrap

            # Should raise ConfigError due to validation failure
            with pytest.raises(ConfigError):
                bootstrap(config_path=str(config_path))

        finally:
            config_path.unlink()

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
            assert llm_resolved.llm.model_name == "gpt-5-mini"

        finally:
            config_path.unlink()

    def test_yaml_with_null_values(self):
        """Test YAML file with null values."""
        config = {"llm": {"api_key": None}, "runtime": {"log_level": None}}

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
            assert llm_resolved.llm.api_key is None

        finally:
            config_path.unlink()

    def test_environment_variable_type_conversion(self):
        """Test that environment variables are properly converted to correct types."""
        test_env = {
            "LOCCA_LLM__TEMPERATURE": "0.8",
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

        # Test with model override
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
        initial_model = initial_llm_config.llm.model_name

        # Call with override
        await runtime.orchestrate("test query", model="gpt-5-mini")

        # Base config should remain unchanged
        updated_llm_config = runtime._llm_manager.config_manager.resolve()
        assert updated_llm_config.llm.model_name == initial_model

    @pytest.mark.asyncio
    async def test_runtime_manager_handles_quota_errors_gracefully(self, ctx):
        """Test that RuntimeManager handles quota exceeded errors gracefully."""
        runtime = ctx.get("runtime")
        # This test will actually make a real API call and may hit quota limits
        # In a real scenario, you might want to mock this or skip based on quota status

        try:
            # Try to make a real API call
            result = await runtime.orchestrate("test query", model="gpt-4.1")

            # If we get here, the API call succeeded
            assert "message" in result
            assert "model_used" in result

        except AgentError as e:
            # If we get a quota error, that's expected and acceptable
            if "insufficient_quota" in str(e) or "quota" in str(e).lower():
                pytest.skip(
                    "OpenAI quota exceeded - skipping test to avoid flaky failures"
                )
            else:
                # Re-raise if it's a different error
                raise
