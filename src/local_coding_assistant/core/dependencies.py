"""Application dependencies container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from local_coding_assistant.core.protocols import IConfigManager

if TYPE_CHECKING:
    from local_coding_assistant.agent.llm.service import LLMService
    from local_coding_assistant.repository.service import RepositoryContextService
    from local_coding_assistant.runtime.events.pipeline import EventPipeline
    from local_coding_assistant.runtime.manager import RuntimeManager
    from local_coding_assistant.sandbox.manager import SandboxManager
    from local_coding_assistant.storage import IExecutionStore
    from local_coding_assistant.tools_platform.gateway import ToolGateway


@dataclass
class AppDependencies:
    """Container for all application dependencies.

    This class holds references to all the core components of the application,
    making them easily accessible throughout the codebase while maintaining
    proper dependency injection.
    """

    config_manager: IConfigManager
    llm_service: LLMService | None = field(default=None)
    tool_gateway: ToolGateway | None = field(default=None)
    runtime_manager: RuntimeManager | None = field(default=None)
    sandbox_manager: SandboxManager | None = field(default=None)
    repository_context_service: RepositoryContextService | None = field(default=None)
    execution_store: IExecutionStore | None = field(default=None)
    event_pipeline: EventPipeline | None = field(default=None)
    _initialized: bool = field(default=False, init=False, repr=False)

    def mark_initialized(self) -> None:
        """Mark dependencies as fully initialized."""
        self._initialized = True

    def is_initialized(self) -> bool:
        """Check if all dependencies are initialized."""
        return self._initialized
