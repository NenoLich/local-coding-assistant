"""Application dependencies container."""

from dataclasses import dataclass, field
from typing import Optional

from local_coding_assistant.agent.llm_manager import LLMManager
from local_coding_assistant.core.protocols import IConfigManager, IToolManager
from local_coding_assistant.runtime.runtime_manager import RuntimeManager


@dataclass
class AppDependencies:
    """Container for all application dependencies.

    This class holds references to all the core components of the application,
    making them easily accessible throughout the codebase while maintaining
    proper dependency injection.
    """

    config_manager: IConfigManager
    llm_manager: LLMManager | None = field(default=None)
    tool_manager: Optional["IToolManager"] = field(default=None)
    runtime_manager: RuntimeManager | None = field(default=None)
    _initialized: bool = field(default=False, init=False, repr=False)

    def mark_initialized(self) -> None:
        """Mark dependencies as fully initialized."""
        self._initialized = True

    def is_initialized(self) -> bool:
        """Check if all dependencies are initialized."""
        return self._initialized
