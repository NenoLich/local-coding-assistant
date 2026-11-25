"""Unit tests for core.protocols module."""

from unittest.mock import Mock

from local_coding_assistant.core.protocols import IConfigManager


class TestIConfigManagerProtocol:
    """Test IConfigManager protocol implementation."""

    def test_protocol_has_required_methods(self):
        """Test that a class implementing IConfigManager has required methods."""

        # This class implements all required methods
        class TestConfigManager:
            @property
            def global_config(self):
                return {}

            @property
            def session_overrides(self):
                return {}

            @property
            def path_manager(self):
                return {}

            @property
            def system_registry(self):
                return {}

            def load_global_config(self):
                return {}

            def get_tools(self):
                return {}

            def register_module(self, module_name, capabilities=None):
                return {}

            def register_capability(self, capabilities):
                return {}

            def unregister_capability(self, capabilities):
                return {}

            def reload_tools(self):
                pass

            def set_session_overrides(self, overrides):
                pass

            def persist_tool_policies_overrides(
                self, rule_id: str, outcome: str, description: str = ""
            ) -> bool:
                pass

            def get_provider_by_model(self, model_id: str) -> str:
                pass

        # This should pass if all required methods are implemented
        assert isinstance(TestConfigManager(), IConfigManager)

    def test_mock_implements_protocol(self):
        """Test that a mock with required methods satisfies the protocol."""
        mock = Mock(spec=IConfigManager)
        assert isinstance(mock, IConfigManager)


def test_iconfigmanager_is_runtime_checkable():
    """Test that IConfigManager is runtime checkable."""

    class BadConfigManager:
        pass

    assert not isinstance(BadConfigManager(), IConfigManager)
