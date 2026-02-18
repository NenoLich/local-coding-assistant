from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from local_coding_assistant.core.protocols import IConfigManager, IToolManager
from local_coding_assistant.runtime.runtime_types import (
    AgentProfile,
    ExecutionMode,
    PromptContext,
    ToolSpec,
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


class ToolSelector:
    """Responsible for exposing tools to the model based on the current mode."""

    def __init__(self, tool_manager: IToolManager | ToolManager | None = None) -> None:
        """Initialize the ToolSelector.

        Args:
            tool_manager: The tool manager instance that provides tool access.
        """
        self.tool_manager = tool_manager

    def select(self, *, tool_call_mode: str) -> tuple[list[ToolSpec], list[str]]:
        """Select tools based on the current execution mode.

        Args:
            tool_call_mode: The current tool call mode ("reasoning_only", "classic", or "ptc")

        Returns:
            A tuple of two lists: (tools, tools_prompt)
        """
        if tool_call_mode == "reasoning_only" or self.tool_manager is None:
            return [], []

        execution_mode = (
            ToolExecutionMode.PTC
            if tool_call_mode == "ptc"
            else ToolExecutionMode.CLASSIC
        )

        tool_specs: list[ToolSpec] = []
        tools = self.tool_manager.list_tools(
            available_only=True, execution_mode=execution_mode
        )
        tools_prompt: list[str] = []

        for entry in tools:
            try:
                resolved = self._resolve_tool_entry(entry)
                if resolved:
                    tool_specs.append(resolved)
            except (ValueError, AttributeError) as e:
                log.warning(f"Skipping invalid tool entry: {e}")
                continue

        if execution_mode == ToolExecutionMode.PTC:
            sandbox_tools = self.tool_manager.list_tools(
                available_only=True, execution_mode=ToolExecutionMode.SANDBOX
            )
            tools_prompt = self.tool_manager.get_sandbox_tools_prompt(sandbox_tools)

        return tool_specs, tools_prompt

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
        handler_context: dict[str, Any] | None = None,
    ) -> PromptContext:
        """Build the context for the LLM based on the current state and configuration.

        Args:
            session: The current session state
            user_input: The user's input
            tool_call_mode: The tool call mode ("reasoning_only", "classic", or "ptc")
            agent_mode: Whether agent mode is enabled
            graph_mode: Whether graph mode is enabled
            handler_context: The handler context to pass to the LLM

        Returns:
            A PromptContext object containing all necessary information for the LLM

        Raises:
            ValueError: If tool_call_mode is invalid or session is not provided
            TypeError: If user_input is not a string
        """
        # Input validation
        if not isinstance(session, SessionState):
            raise ValueError("session must be an instance of SessionState")

        if not isinstance(tool_call_mode, str):
            raise TypeError(
                f"tool_call_mode must be a string, got {type(tool_call_mode).__name__}"
            )

        tool_call_mode = tool_call_mode.lower()

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

        if runtime_config.tool_call_mode != tool_call_mode:
            self.config_manager.set_session_overrides(
                {"runtime.tool_call_mode": tool_call_mode}
            )

        # Map validated tool_call_mode to execution_mode
        execution_mode = self._map_validated_mode_to_execution_mode(tool_call_mode)

        agent = self._resolve_agents(agent_mode=agent_mode, graph_mode=graph_mode)
        memories = self.memory_provider.fetch(session=session)
        skills = self.skill_provider.resolve(execution_mode=execution_mode)
        tools, tools_prompt = self.tool_selector.select(tool_call_mode=tool_call_mode)

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
            "Context built: execution_mode=%s agent=%s tools=%d",
            execution_mode,
            agent.name if agent else "none",
            len(tools),
        )

        return PromptContext(
            session_id=session.id,
            execution_mode=execution_mode,
            tool_call_mode=tool_call_mode,
            user_input=user_input,
            agent_profile=agent,  # Single profile instead of list
            active_skills=skills,
            tools_prompt=tools_prompt,
            memories=memories,
            tools=tools,
            history=[m.model_dump() for m in session.history],
            metadata=metadata,
            is_sandbox_enabled=sandbox_config.enabled,
            handler_context=handler_context,
        )

    def _map_validated_mode_to_execution_mode(
        self, tool_call_mode: str
    ) -> ExecutionMode:
        """Map a validated tool_call_mode to execution_mode.

        Args:
            tool_call_mode: The validated tool call mode

        Returns:
            The corresponding ExecutionMode
        """
        if tool_call_mode == "reasoning_only":
            return ExecutionMode.REASONING_ONLY
        elif tool_call_mode == "ptc":
            return ExecutionMode.SANDBOX_PYTHON
        else:  # "classic" or any other valid mode
            return ExecutionMode.CLASSIC_TOOLS

    def _resolve_agents(
        self, *, agent_mode: bool, graph_mode: bool
    ) -> AgentProfile | None:
        """Resolve which agent profile to use based on the current mode.

        Args:
            agent_mode: Whether agent mode is enabled
            graph_mode: Whether graph mode is enabled

        Returns:
            Single AgentProfile to use for the current context
        """
        # Use explicitly provided profile if available
        if self._agent_catalog:
            return self._agent_catalog[0]  # Return first (and only) profile

        # Get agent config
        agent_config = self.config_manager.global_config.agent

        # In graph mode, use the planner profile (executor would be separate agent)
        if graph_mode:
            planner = agent_config.get_profile("planner")
            return AgentProfile(**planner.model_dump())

        # In agent mode, use the default agent profile
        if agent_mode:
            default_profile = agent_config.get_profile("default")
            return AgentProfile(**default_profile.model_dump())

        # Fall back to default profile if no specific mode is set

        default_profile = agent_config.get_profile("default")
        return AgentProfile(**default_profile.model_dump())


__all__ = [
    "AgentProfile",
    "ContextManager",
    "MemoryProvider",
    "PromptContext",
    "SkillProvider",
    "ToolSelector",
]
