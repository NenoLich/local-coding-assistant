"""Abstract base class for sandbox implementations."""

from abc import ABC, abstractmethod

from .sandbox_types import SandboxExecutionRequest, SandboxExecutionResponse


class ISandbox(ABC):
    """Interface for sandbox implementations."""

    @abstractmethod
    async def start(self) -> None:
        """Start the sandbox environment."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the sandbox environment."""
        pass

    @abstractmethod
    async def stop_session(self, session_id: str) -> None:
        """Stop a specific session."""
        pass

    @abstractmethod
    async def execute(
        self, request: SandboxExecutionRequest
    ) -> SandboxExecutionResponse:
        """Execute code in the sandbox.

        Args:
            request: The execution request containing code and metadata.

        Returns:
            The execution response.
        """
        pass

    async def __aenter__(self):
        """Enter the runtime context related to this object."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context related to this object."""
        await self.stop()

    @abstractmethod
    def check_availability(self) -> bool:
        """Check if the sandbox environment is available.

        Returns:
            bool: True if the sandbox is available, False otherwise
        """
        pass
