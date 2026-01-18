from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from local_coding_assistant.core.protocols import IConfigManager
from local_coding_assistant.runtime.runtime_types import AgentProfile
from local_coding_assistant.runtime.session import SessionState
from local_coding_assistant.tools.tool_manager import ToolManager


# ─── MOCK CLASSES ────────────────────────────────────────────────────────────


class MockTool:
    """Mock tool class for testing tool-related functionality."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Optional[dict] = None,
        execution_mode: str = "classic",
        available: bool = True,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters or {
            "type": "object",
            "properties": {},
            "required": [],
        }
        self.execution_mode = execution_mode
        self.available = available
        self.category = None  # Add category attribute
        self.source = None  # Add source attribute
        self.tags = []  # Add tags attribute
        self.permissions = []  # Add permissions attribute

    def __str__(self):
        return f"MockTool(name={self.name}, description={self.description})"

    def to_dict(self):
        """Convert tool to dictionary format expected by the ToolSelector."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "execution_mode": self.execution_mode,
            "available": self.available,
            "category": self.category,
            "source": self.source,
            "tags": self.tags,
            "permissions": self.permissions,
        }


class MockToolManager:
    """Mock tool manager for testing tool selection and execution."""

    def __init__(self, tools: Optional[list] = None):
        self.tools = tools or []

    def list_tools(
        self, available_only: bool = True, execution_mode: Any = None
    ) -> list:
        filtered_tools = self.tools

        # Filter by availability
        if available_only:
            filtered_tools = [
                t for t in filtered_tools if getattr(t, "available", True)
            ]

        # Filter by execution mode
        if execution_mode is not None:
            filtered_tools = [
                t
                for t in filtered_tools
                if getattr(t, "execution_mode", "classic") == execution_mode
            ]

        return filtered_tools

    def has_runtime(self, runtime_name: str) -> bool:
        return runtime_name == "execute_python_code"


@dataclass
class AttrDict:
    """A dictionary that supports both attribute and item access."""

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], dict):
            self.__dict__.update(args[0])
        else:
            self.__dict__.update(kwargs)

    def __getattr__(self, key):
        try:
            return self.__dict__[key]
        except KeyError as e:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{key}'"
            ) from e

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __delattr__(self, key):
        if key in self.__dict__:
            del self.__dict__[key]

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def update(self, other):
        self.__dict__.update(other)

    def __contains__(self, key):
        return key in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)

    def items(self):
        return self.__dict__.items()

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()


class MockConfigManager(IConfigManager):
    """Mock configuration manager for testing configuration-related functionality."""

    def __init__(self):
        # Initialize _config as the main configuration store
        self._config = AttrDict(
            {
                "runtime": AttrDict(
                    {
                        "persistent_sessions": False,
                        "use_graph_mode": False,
                        "stream": True,
                        "tool_call_mode": "classic",  # Changed from "auto" to "classic"
                    }
                ),
                "llm": AttrDict(
                    {
                        "model_name": "test-model",
                        "provider": "test",
                        "temperature": 0.7,
                        "max_tokens": 1000,
                    }
                ),
                "sandbox": AttrDict({"enabled": True}),
                "tools": AttrDict({"tools": []}),
                "providers": AttrDict({}),
                "agent": AttrDict(
                    {
                        "get_profile": lambda _: AgentProfile(
                            name="default", description="Default agent"
                        )
                    }
                ),
                # Prompt configuration with required settings
                "prompt": AttrDict(
                    {
                        "templates": {
                            "classic": "classic_template.txt",
                            "reasoning_only": "reasoning_only_template.txt",
                            "ptc": "ptc_template.txt",
                        },
                        "enable_jinja_autoescape": True,
                        "trim_blocks": True,
                        "lstrip_blocks": True,
                        "template_dirs": ["path/to/templates"],
                        "default_template": "classic",
                    }
                ),
            }
        )
        self._session_overrides = {}
        self._global_config = self._config  # Make _global_config point to _config

    @property
    def global_config(self):
        return self._global_config

    @global_config.setter
    def global_config(self, value):
        self._global_config = value
        # Also update _config if needed
        if hasattr(self, "_config"):
            self._config = value

    @property
    def session_overrides(self):
        return self._session_overrides

    def set_session_overrides(self, overrides: dict) -> None:
        self._session_overrides.update(overrides)

    def resolve(self, *args, **kwargs):
        return self._global_config

    def get_tools(self):
        # Return a dictionary of tools that can be iterated with .items()
        return {}

    def list_tools(self):
        # Return a list of tools for compatibility
        return []

    def reload_tools(self):
        pass


class ToolManagerHelper(ToolManager):
    """Test tool manager that supports invoke() for backward compatibility."""

    def __init__(self, config_manager: IConfigManager):
        super().__init__(config_manager=config_manager)
        self._tools = {}

    def add_tool(self, name: str, tool: Any) -> None:
        """Add a tool to the manager."""
        self._tools[name] = tool

    def list_tools(
        self,
        available_only: bool = True,
        execution_mode: str | Any = None,
        category: str | Any = None,
    ) -> list:
        """List tools with the given filters."""
        tools = list(self._tools.values())

        if available_only:
            tools = [t for t in tools if getattr(t, "available", True)]

        if execution_mode:
            tools = [
                t for t in tools if getattr(t, "execution_mode", None) == execution_mode
            ]

        return tools

    # Legacy invoke method for backward compatibility
    async def invoke(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Tool {name} not found")
        return await tool(payload)


# ─── FIXTURES ────────────────────────────────────────────────────────────────


@pytest.fixture
def config_manager() -> MockConfigManager:
    """Fixture providing a configured mock config manager."""
    return MockConfigManager()


@pytest.fixture
def test_tool():
    """Fixture providing a test tool with all required attributes."""
    tool = MagicMock()
    tool.name = "test_tool"
    tool.description = "Test tool"
    tool.parameters = {"type": "object", "properties": {"param1": {"type": "string"}}}
    tool.execution_mode = "classic"
    tool.available = True
    tool.category = "test"
    tool.is_available = True
    tool.is_async = False
    tool.args_schema = {"type": "object", "properties": {"param1": {"type": "string"}}}
    tool.return_value = {"result": "test result"}
    return tool


@pytest.fixture
def tool_manager(config_manager: MockConfigManager, test_tool):
    """Fixture providing a mock tool manager with test tools."""
    # Create a MockToolManager with a test tool
    manager = MockToolManager(tools=[test_tool])
    return manager


@pytest.fixture
def session_state() -> SessionState:
    """Fixture providing a test session state."""
    return SessionState(id="test_session", history=[], metadata={"test_meta": "value"})


# Re-export for backward compatibility
__all__ = [
    "MockTool",
    "MockToolManager",
    "MockConfigManager",
    "ToolManagerHelper",
    "config_manager",
    "tool_manager",
    "session_state",
]
