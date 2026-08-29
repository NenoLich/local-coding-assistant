from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from local_coding_assistant.core.protocols import IConfigManager, IToolManager
from local_coding_assistant.core.telemetry_types import FileChange, FileChangeType
from local_coding_assistant.repository import (
    ProjectInfo,
    RepoMapData,
    RepositoryContextService,
)
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


@dataclass(frozen=True)
class ToolsBundle:
    """Represents the state of available tools for caching."""

    tool_call_mode: str
    tool_names: tuple[str, ...]  # Sorted list of available tool names
    execution_mode: ToolExecutionMode

    @classmethod
    def from_tool_manager(
        cls, tool_manager: IToolManager | None, tool_call_mode: str
    ) -> ToolsBundle:
        """Create bundle from current tool manager state."""
        execution_mode = (
            ToolExecutionMode.PTC
            if tool_call_mode == "ptc"
            else ToolExecutionMode.CLASSIC
        )
        if tool_manager is None:
            tool_names = ()
        else:
            tools = tool_manager.list_tools(
                available_only=True, execution_mode=execution_mode
            )
            tool_names = tuple(
                sorted(
                    [str(t[0]) if isinstance(t, tuple) else str(t.name) for t in tools]
                )
            )
        return cls(
            tool_call_mode=tool_call_mode,
            tool_names=tool_names,
            execution_mode=execution_mode,
        )


class StaticComponentCache:
    def __init__(
        self,
        config_manager: IConfigManager,
        tool_manager: IToolManager | ToolManager | None = None,
        skill_provider: SkillProvider | None = None,
        agent_profiles: Iterable[AgentProfile] | None = None,
        repository_context_service: RepositoryContextService | None = None,
    ):
        self.config_manager = config_manager
        self.tool_selector = ToolSelector(tool_manager=tool_manager)
        self.skill_provider = skill_provider or SkillProvider()
        self.repository_context_service = repository_context_service
        self._agent_catalog = list(agent_profiles) if agent_profiles else []

        # Caches
        self._agent_profile_cache: dict[tuple[bool, bool], AgentProfile] = {}
        self._tools_cache: dict[ToolsBundle, tuple[list[ToolSpec], list[str]]] = {}
        self._skills_cache: dict[str, list[str]] = {}
        self._repo_context_cache: tuple[ProjectInfo, RepoMapData | None] | None = None

    def get_agent_profile(self, agent_mode: bool, graph_mode: bool) -> AgentProfile:
        """Get cached agent profile or compute if not cached."""
        cache_key = (agent_mode, graph_mode)
        if cache_key in self._agent_profile_cache:
            log.debug(
                "Agent profile cache hit for agent_mode=%s, graph_mode=%s",
                agent_mode,
                graph_mode,
            )
            return self._agent_profile_cache[cache_key]

        log.debug(
            "Agent profile cache miss for agent_mode=%s, graph_mode=%s",
            agent_mode,
            graph_mode,
        )

        # Use explicitly provided profile if available
        if self._agent_catalog:
            profile = self._agent_catalog[0]  # Return first (and only) profile
        else:
            # Get agent config
            agent_config = self.config_manager.global_config.agent

            # In graph mode, use the planner profile (executor would be separate agent)
            if graph_mode:
                profile_data = agent_config.get_profile("planner")
                profile = AgentProfile(**profile_data.model_dump())
            # In agent mode, use the default agent profile
            elif agent_mode:
                profile_data = agent_config.get_profile("default")
                profile = AgentProfile(**profile_data.model_dump())
            # Fall back to default profile if no specific mode is set
            else:
                profile_data = agent_config.get_profile("default")
                profile = AgentProfile(**profile_data.model_dump())

        self._agent_profile_cache[cache_key] = profile
        return profile

    def get_tools(
        self, tool_call_mode: str, tool_manager: IToolManager | None
    ) -> tuple[list[ToolSpec], list[str]]:
        """Get cached tools or compute if not cached."""
        bundle = ToolsBundle.from_tool_manager(tool_manager, tool_call_mode)
        if bundle in self._tools_cache:
            log.debug("Tools cache hit for bundle: %s", bundle)
            return self._tools_cache[bundle]

        log.debug("Tools cache miss for bundle: %s", bundle)
        tools, tools_prompt = self.tool_selector.select(tool_call_mode=tool_call_mode)
        self._tools_cache[bundle] = (tools, tools_prompt)
        return tools, tools_prompt

    def get_skills(self, execution_mode: str) -> list[str]:
        """Get cached skills or compute if not cached."""
        if execution_mode in self._skills_cache:
            log.debug("Skills cache hit for execution_mode=%s", execution_mode)
            return self._skills_cache[execution_mode]

        log.debug("Skills cache miss for execution_mode=%s", execution_mode)
        skills = self.skill_provider.resolve(execution_mode=execution_mode)
        self._skills_cache[execution_mode] = skills
        return skills

    def get_repo_context(self) -> tuple[ProjectInfo, RepoMapData | None, bool] | None:
        """Get cached repository context or compute if not cached."""
        is_fresh = False
        if self._repo_context_cache is not None:
            log.debug("Repo context cache hit")
            proj_info, repo_map = self._repo_context_cache
            return proj_info, repo_map, is_fresh

        log.debug("Repo context cache miss")
        if self.repository_context_service is None:
            log.debug("Repository context service not available")
            return None

        try:
            proj_info, repo_map = self.repository_context_service.get_repo_context()
            is_fresh = True
            self._repo_context_cache = proj_info, repo_map
            return proj_info, repo_map, is_fresh
        except Exception as e:
            log.warning(f"Failed to get repository context: {e}")
            return None

    def invalidate_all(self):
        """Clear all caches."""
        log.info("Invalidated all caches")
        self._agent_profile_cache.clear()
        self._tools_cache.clear()
        self._skills_cache.clear()
        self._repo_context_cache = None


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
        agent_profiles: Iterable[AgentProfile] | None = None,
        repository_context_service: RepositoryContextService | None = None,
    ) -> None:
        self.config_manager = config_manager
        self.tool_manager = tool_manager
        self.repository_context_service = repository_context_service
        self.memory_provider = memory_provider or MemoryProvider()
        self._cache = StaticComponentCache(
            config_manager=config_manager,
            tool_manager=tool_manager,
            skill_provider=skill_provider,
            agent_profiles=agent_profiles,
            repository_context_service=repository_context_service,
        )
        self.user_file_changes: list[FileChange] = []

        # Register file monitoring callback if repository service is available
        if (
            repository_context_service
            and repository_context_service.file_monitoring_service
        ):
            repository_context_service.register_file_change_callback(
                self._on_file_change
            )

    def _on_file_change(self, event: Any) -> None:
        """Handle file change events from file monitoring service.

        Args:
            event: FileChangeEvent from file monitoring service.
        """
        # Convert FileChangeEvent to FileChange
        from local_coding_assistant.repository.file_monitoring import FileChangeEvent
        from local_coding_assistant.repository.file_monitoring import (
            FileChangeType as RepoFileChangeType,
        )

        if isinstance(event, FileChangeEvent):
            # Map repository FileChangeType to telemetry FileChangeType
            change_type_map = {
                RepoFileChangeType.MODIFIED: FileChangeType.MODIFIED,
                RepoFileChangeType.CREATED: FileChangeType.CREATED,
                RepoFileChangeType.DELETED: FileChangeType.DELETED,
            }
            change_type = change_type_map.get(
                event.change_type, FileChangeType.MODIFIED
            )
            file_change = FileChange(path=event.path, change_type=change_type)
            self.user_file_changes.append(file_change)

    def flush_user_file_changes(self) -> None:
        """Clear user file changes after they have been processed."""
        if self.user_file_changes:
            log.debug(f"Flushing {len(self.user_file_changes)} user file changes")
            self.user_file_changes.clear()

    def _format_file_changes_message(self, file_changes: list[FileChange]) -> str:
        """Format file changes into a system message.

        Args:
            file_changes: List of file changes to format.

        Returns:
            Formatted system message string.
        """
        if not file_changes:
            return ""

        lines = [
            "The user made recent changes to the filesystem outside of this conversation. Changes:"
        ]
        for change in file_changes:
            lines.append(f"- [{change.change_type.value.capitalize()}] {change.path}")
        lines.append(
            "If some files are relevant to the current task they need to be rediscovered. Do not respond to this message."
        )
        return "\n".join(lines)

    def build_context(
        self,
        *,
        session: SessionState,
        user_input: str,
        tool_call_mode: str,
        agent_mode: bool = False,
        graph_mode: bool = False,
        handler_context: dict[str, Any] | None = None,
        agent_file_changes: list[FileChange] | None = None,
    ) -> PromptContext:
        """Build the context for the LLM based on the current state and configuration.

        Args:
            session: The current session state
            user_input: The user's input
            tool_call_mode: The tool call mode ("reasoning_only", "classic", or "ptc")
            agent_mode: Whether agent mode is enabled
            graph_mode: Whether graph mode is enabled
            handler_context: The handler context to pass to the LLM
            agent_file_changes: List of file changes made by the agent in previous turn

        Returns:
            A PromptContext object containing all necessary information for the LLM

        Raises:
            ValueError: If tool_call_mode is invalid or session is not provided
            TypeError: If user_input is not a string
        """
        tool_call_mode = tool_call_mode.lower()
        runtime_config = self.config_manager.global_config.runtime
        sandbox_config = self.config_manager.global_config.sandbox

        if runtime_config.tool_call_mode != tool_call_mode:
            self.config_manager.set_session_overrides(
                {"runtime.tool_call_mode": tool_call_mode}
            )

        # Map validated tool_call_mode to execution_mode
        execution_mode = self._map_validated_mode_to_execution_mode(tool_call_mode)

        # Get cached static components
        agent = self._cache.get_agent_profile(agent_mode, graph_mode)
        tools, tools_prompt = self._cache.get_tools(tool_call_mode, self.tool_manager)
        skills = self._cache.get_skills(execution_mode)

        project_info = None
        repo_map_data = None
        is_repo_context_fresh = None
        repo_context = self._cache.get_repo_context()
        if repo_context is not None:
            project_info, repo_map_data, is_repo_context_fresh = repo_context

        if is_repo_context_fresh is not None and not is_repo_context_fresh:
            # Combine and filter file changes
            combined_file_changes: list[FileChange] = []
            agent_file_paths = (
                {fc.path for fc in agent_file_changes} if agent_file_changes else set()
            )

            # Add user file changes, excluding those made by agent
            for user_change in self.user_file_changes:
                if user_change.path not in agent_file_paths:
                    combined_file_changes.append(user_change)

            # Format and append file changes to session history
            if combined_file_changes:
                file_change_message = self._format_file_changes_message(
                    combined_file_changes
                )
                session.add_system_message(file_change_message)

        # Flush user file changes after processing
        self.flush_user_file_changes()

        # Dynamic components (always fresh)
        memories = self.memory_provider.fetch(session=session)
        history = [m.model_dump() for m in session.history]

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
            agent_profile=agent,
            active_skills=skills,
            tools_prompt=tools_prompt,
            memories=memories,
            tools=tools,
            history=history,
            metadata=metadata,
            is_sandbox_enabled=sandbox_config.enabled,
            handler_context=handler_context,
            project_info=project_info,
            repo_map_data=repo_map_data,
        )

    def invalidate_all(self):
        """Invalidate all caches."""
        self._cache.invalidate_all()

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


__all__ = [
    "AgentProfile",
    "ContextManager",
    "MemoryProvider",
    "PromptContext",
    "SkillProvider",
    "ToolSelector",
]
