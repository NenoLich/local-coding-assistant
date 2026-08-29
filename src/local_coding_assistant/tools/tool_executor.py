"""Tool executor singleton for repository operations."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from local_coding_assistant.repository.service import RepositoryContextService


class ToolExecutor:
    """Singleton executor for repository operations accessible by tools."""

    _instance: "ToolExecutor | None" = None
    _repository_service: "RepositoryContextService | None" = None

    def __new__(cls) -> "ToolExecutor":
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def set_repository_service(
        cls, repository_service: "RepositoryContextService"
    ) -> None:
        """Set the repository service.

        Args:
            repository_service: The repository context service instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        cls._instance._repository_service = repository_service

    @classmethod
    def get_instance(cls) -> "ToolExecutor":
        """Get the singleton instance.

        Returns:
            The ToolExecutor instance.

        Raises:
            RuntimeError: If the executor has not been initialized.
        """
        if cls._instance is None:
            raise RuntimeError("ToolExecutor has not been initialized")
        return cls._instance

    def search_symbols(
        self,
        query: str,
        symbol_types: list[str] | None = None,
        file_path: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Search for symbols across the codebase.

        Args:
            query: Search query for symbol names.
            symbol_types: Optional filter by symbol types (function, class, method, etc.).
            file_path: Optional filter by specific file path.
            limit: Maximum number of results to return.

        Returns:
            List of matching symbols as dictionaries.

        Raises:
            RuntimeError: If repository service is not initialized.
        """
        if self._repository_service is None:
            raise RuntimeError("Repository service not initialized")

        results = self._repository_service.search_symbols(
            query, symbol_types, file_path, limit
        )

        return [r.model_dump() for r in results]


def get_tool_executor() -> ToolExecutor:
    """Get the tool executor singleton instance.

    Returns:
        The ToolExecutor instance.

    Raises:
        RuntimeError: If the executor has not been initialized.
    """
    return ToolExecutor.get_instance()
