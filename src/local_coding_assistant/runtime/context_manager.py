from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from typing_extensions import runtime_checkable

from local_coding_assistant.core.protocols import IConfigManager, IToolManager
from local_coding_assistant.runtime.runtime_types import (
    AgentProfile,
    ExecutionMode,
    PromptContext,
)
from local_coding_assistant.runtime.session import SessionState
from local_coding_assistant.tools.types import ToolExecutionMode
from local_coding_assistant.utils.logging import get_logger

if TYPE_CHECKING:
    from local_coding_assistant.tools.tool_manager import ToolManager


log = get_logger("runtime.context_manager")


class MemoryProvider:
    """Placeholder memory retrieval component."""

    def fetch(self, *, session: SessionState) -> list[str]:
        log.debug("MemoryProvider.fetch called (placeholder)")
        return []


class SkillProvider:
    """Placeholder skill-selection component."""

    def resolve(self, *, execution_mode: str) -> list[str]:
        log.debug(
            "SkillProvider.resolve called for mode %s (placeholder)", execution_mode
        )
        return []


@runtime_checkable
class ToolProtocol(Protocol):
    """Protocol defining the expected interface for tool objects."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolSpec:
    """Data class representing a tool specification."""

    name: str
    description: str
    parameters: dict[str, Any]


class ToolSelector:
    """Responsible for exposing tools to the model based on the current mode."""

    def __init__(self, tool_manager: IToolManager | ToolManager | None = None) -> None:
        """Initialize the ToolSelector.

        Args:
            tool_manager: The tool manager instance that provides tool access.
        """
        self.tool_manager = tool_manager

    def select(self, *, tool_call_mode: str) -> list[dict[str, Any]]:
        """Select tools based on the current execution mode.

        Args:
            tool_call_mode: The current tool call mode ("reasoning_only", "classic", or "ptc")

        Returns:
            A list of tool specifications in the required format.
        """
        if tool_call_mode == "reasoning_only" or self.tool_manager is None:
            return []

        execution_mode = (
            ToolExecutionMode.PTC
            if tool_call_mode == "ptc"
            else ToolExecutionMode.CLASSIC
        )

        tool_specs: list[dict[str, Any]] = []
        for entry in self.tool_manager.list_tools(
            available_only=True, execution_mode=execution_mode
        ):
            try:
                resolved = self._resolve_tool_entry(entry)
                if resolved:
                    tool_specs.append(self._build_tool_spec(resolved))
            except (ValueError, AttributeError) as e:
                log.warning(f"Skipping invalid tool entry: {e}")
                continue

        return tool_specs

    def _resolve_tool_entry(self, entry: Any) -> ToolSpec | None:
        """Resolve a tool entry into a standardized ToolSpec.

        Args:
            entry: The tool entry to resolve. Can be a ToolSpec, a tuple of (name, tool),
                  or any object with name and description attributes.

        Returns:
            A ToolSpec instance if the entry is valid, None otherwise.

        Raises:
            ValueError: If the entry is invalid or missing required attributes.
            AttributeError: If the entry is malformed.
        """
        if entry is None:
            raise ValueError("Tool entry cannot be None")

        name, tool_obj = self._extract_name_and_tool(entry)
        description = self._extract_description(tool_obj, name)
        parameters = self._extract_parameters(tool_obj, name)

        return ToolSpec(name=name, description=description, parameters=parameters)

    def _extract_name_and_tool(self, entry: Any) -> tuple[str, Any]:
        """Extract name and tool object from the entry.

        Args:
            entry: The tool entry to process

        Returns:
            A tuple of (name, tool_object)

        Raises:
            ValueError: If the entry format is invalid
        """
        if isinstance(entry, tuple) and len(entry) == 2:
            name, tool_obj = entry
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Tool name in tuple must be a non-empty string")
            if tool_obj is None:
                raise ValueError("Tool object in tuple cannot be None")
            return name, tool_obj

        tool_obj = entry
        name = self._determine_tool_name(tool_obj)
        return name, tool_obj

    def _determine_tool_name(self, tool_obj: Any) -> str:
        """Determine the tool name from the tool object.

        Args:
            tool_obj: The tool object to extract name from

        Returns:
            The tool name

        Raises:
            ValueError: If the name cannot be determined
        """
        if hasattr(tool_obj, "name"):
            name = tool_obj.name
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Tool name must be a non-empty string")
            return name

        name = str(tool_obj)
        if not name.strip():
            raise ValueError("Could not determine tool name from entry")
        return name

    def _extract_description(self, tool_obj: Any, name: str) -> str:
        """Extract and validate description from the tool object.

        Args:
            tool_obj: The tool object to extract description from
            name: The tool name for error messages

        Returns:
            The tool description

        Raises:
            ValueError: If description is missing or invalid
        """
        if not hasattr(tool_obj, "description"):
            raise ValueError(
                f"Tool '{name}' is missing required 'description' attribute"
            )

        description = getattr(tool_obj, "description", "")
        if not isinstance(description, str):
            raise ValueError(f"Tool '{name}' description must be a string")

        return description

    def _extract_parameters(self, tool_obj: Any, name: str) -> dict[str, Any]:
        """Extract and validate parameters from the tool object.

        Args:
            tool_obj: The tool object to extract parameters from
            name: The tool name for error messages

        Returns:
            The tool parameters dictionary

        Raises:
            ValueError: If parameters are invalid
        """
        parameters = {}
        if hasattr(tool_obj, "parameters") and tool_obj.parameters is not None:
            if not isinstance(tool_obj.parameters, dict):
                raise ValueError(f"Tool '{name}' parameters must be a dictionary")
            parameters = dict(tool_obj.parameters)

        if not parameters:
            parameters = {"type": "object", "properties": {}, "required": []}

        return parameters

    def _build_tool_spec(self, tool_spec: ToolSpec) -> dict[str, Any]:
        """Build the final tool specification dictionary.

        Args:
            tool_spec: The ToolSpec to convert to a dictionary

        Returns:
            A dictionary representing the tool specification
        """
        return {
            "type": "function",
            "function": {
                "name": tool_spec.name,
                "description": tool_spec.description,
                "parameters": tool_spec.parameters,
            },
        }


class ContextManager:
    """High-level orchestrator that decides what the LLM should see."""

    def __init__(
        self,
        config_manager: IConfigManager,
        *,
        tool_manager: IToolManager | ToolManager | None = None,
        memory_provider: MemoryProvider | None = None,
        skill_provider: SkillProvider | None = None,
        tool_selector: ToolSelector | None = None,
        agent_profiles: Iterable[AgentProfile] | None = None,
    ) -> None:
        self.config_manager = config_manager
        self.tool_selector = tool_selector or ToolSelector(tool_manager=tool_manager)
        self.memory_provider = memory_provider or MemoryProvider()
        self.skill_provider = skill_provider or SkillProvider()
        self._agent_catalog = list(agent_profiles) if agent_profiles else []

    def build_context(
        self,
        *,
        session: SessionState,
        user_input: str,
        tool_call_mode: str,
        agent_mode: bool = False,
        graph_mode: bool = False,
    ) -> PromptContext:
        """Build the context for the LLM based on the current state and configuration.

        Args:
            session: The current session state
            user_input: The user's input
            tool_call_mode: The tool call mode ("reasoning_only", "classic", or "ptc")
            agent_mode: Whether agent mode is enabled
            graph_mode: Whether graph mode is enabled

        Returns:
            A PromptContext object containing all necessary information for the LLM

        Raises:
            ValueError: If tool_call_mode is invalid or session is not provided
            TypeError: If user_input is not a string
        """
        # Input validation
        if not isinstance(session, SessionState):
            raise ValueError("session must be an instance of SessionState")

        if not isinstance(user_input, str):
            raise TypeError(
                f"user_input must be a string, got {type(user_input).__name__}"
            )

        if not isinstance(tool_call_mode, str):
            raise TypeError(
                f"tool_call_mode must be a string, got {type(tool_call_mode).__name__}"
            )

        tool_call_mode = tool_call_mode.lower()
        valid_modes = {"reasoning_only", "classic", "ptc"}
        if tool_call_mode not in valid_modes:
            raise ValueError(
                f"Invalid tool_call_mode: '{tool_call_mode}'. "
                f"Must be one of: {', '.join(sorted(valid_modes))}"
            )

        if not isinstance(agent_mode, bool):
            raise TypeError(
                f"agent_mode must be a boolean, got {type(agent_mode).__name__}"
            )

        if not isinstance(graph_mode, bool):
            raise TypeError(
                f"graph_mode must be a boolean, got {type(graph_mode).__name__}"
            )
        runtime_config = self.config_manager.global_config.runtime
        sandbox_config = self.config_manager.global_config.sandbox

        execution_mode = self._resolve_execution_mode(
            tool_call_mode, sandbox_config.enabled
        )
        agents = self._resolve_agents(agent_mode=agent_mode, graph_mode=graph_mode)
        memories = self.memory_provider.fetch(session=session)
        skills = self.skill_provider.resolve(execution_mode=execution_mode)
        tools = self.tool_selector.select(tool_call_mode=tool_call_mode)

        metadata = {
            "execution_mode": execution_mode,
            "tool_call_mode": tool_call_mode,
            "graph_mode": graph_mode,
            "agent_mode": agent_mode,
            "streaming": runtime_config.stream,
            "sandbox_enabled": sandbox_config.enabled,
        }
        metadata.update(session.metadata or {})

        log.debug(
            "Context built: execution_mode=%s agents=%d tools=%d",
            execution_mode,
            len(agents),
            len(tools),
        )

        return PromptContext(
            session_id=session.id,
            execution_mode=execution_mode,
            tool_call_mode=tool_call_mode,
            user_input=user_input,
            agent_profiles=agents,
            active_skills=skills,
            memories=memories,
            tools=tools,
            history=[m.model_dump() for m in session.history],
            metadata=metadata,
            is_sandbox_enabled=sandbox_config.enabled,
        )

    def _resolve_execution_mode(
        self, tool_call_mode: str, sandbox_enabled: bool
    ) -> ExecutionMode:
        """Resolve the execution mode based on tool call mode and sandbox availability.

        Args:
            tool_call_mode: The requested tool call mode ("reasoning_only", "classic", or "ptc")
            sandbox_enabled: Whether sandbox execution is enabled in the configuration

        Returns:
            The resolved ExecutionMode

        Raises:
            ValueError: If the tool_call_mode is invalid
        """
        tool_call_mode = tool_call_mode.lower()
        valid_modes = {"reasoning_only", "classic", "ptc"}
        if tool_call_mode not in valid_modes:
            raise ValueError(
                f"Invalid tool_call_mode: '{tool_call_mode}'. "
                f"Must be one of: {', '.join(valid_modes)}"
            )

        if tool_call_mode == "reasoning_only":
            log.info("Execution mode set to REASONING_ONLY (explicitly requested)")
            return ExecutionMode.REASONING_ONLY

        # Check for tool manager availability
        if (
            not hasattr(self.tool_selector, "tool_manager")
            or not self.tool_selector.tool_manager
        ):
            log.warning(
                "No tool manager available. Tool-based execution modes require a valid tool manager. "
                "Falling back to REASONING_ONLY mode."
            )
            return ExecutionMode.REASONING_ONLY

        # Handle PTC (Python Tool Calling) mode
        if tool_call_mode == "ptc":
            if not sandbox_enabled:
                log.warning(
                    "PTC mode requires sandbox to be enabled. Sandbox is currently disabled in configuration. "
                    "Falling back to CLASSIC_TOOLS mode."
                )
                return ExecutionMode.CLASSIC_TOOLS

            if not self.tool_selector.tool_manager.has_runtime("execute_python_code"):
                log.warning(
                    "PTC mode requires 'execute_python_code' runtime, which is not available. "
                    "Please ensure the Python execution environment is properly configured. "
                    "Falling back to CLASSIC_TOOLS mode."
                )
                return ExecutionMode.CLASSIC_TOOLS

            log.info("Execution mode set to SANDBOX_PYTHON (PTC mode with sandbox)")
            return ExecutionMode.SANDBOX_PYTHON

        # Default to CLASSIC_TOOLS for any other valid tool_call_mode
        log.info("Execution mode set to CLASSIC_TOOLS")
        return ExecutionMode.CLASSIC_TOOLS

    def _resolve_agents(
        self, *, agent_mode: bool, graph_mode: bool
    ) -> list[AgentProfile]:
        """Resolve which agent profiles to use based on the current mode.

        Args:
            agent_mode: Whether agent mode is enabled
            graph_mode: Whether graph mode is enabled

        Returns:
            List of AgentProfile instances to use for the current context
        """
        # Use explicitly provided profiles if available
        if self._agent_catalog:
            return list(self._agent_catalog)

        # Get agent config
        agent_config = self.config_manager.global_config.agent

        # In graph mode, use the planner and executor profiles
        if graph_mode:
            planner = agent_config.get_profile("planner")
            executor = agent_config.get_profile("executor")
            return [
                AgentProfile(**planner.model_dump()),
                AgentProfile(**executor.model_dump()),
            ]

        # In agent mode, use the default agent profile
        if agent_mode:
            default_profile = agent_config.get_profile("default")
            return [AgentProfile(**default_profile.model_dump())]

        # Fall back to default profile if no specific mode is set
        default_profile = agent_config.get_profile("default")
        return [AgentProfile(**default_profile.model_dump())]


__all__ = [
    "AgentProfile",
    "ContextManager",
    "MemoryProvider",
    "PromptContext",
    "SkillProvider",
    "ToolSelector",
]
