"""Application context holding shared resources and state."""

from typing import Any


class AppContext:
    """Central context object that holds application-wide resources."""

    def __init__(self) -> None:
        """Initialize the application context with default values."""
        self._resources: dict[str, Any] = {}

    def register(self, name: str, resource: Any) -> None:
        """Register a resource with the context.

        Args:
            name: Unique name for the resource
            resource: The resource object to store
        """
        self._resources[name] = resource

    def get(self, name: str, default: Any = None) -> Any | None:
        """Retrieve a resource by name.

        Args:
            name: Name of the resource to retrieve
            default: Default value if resource not found

        Returns:
            The requested resource or default if not found
        """
        return self._resources.get(name, default)

    def __getitem__(self, name: str) -> Any:
        """Enable dictionary-style access to resources.

        Raises:
            KeyError: If the resource is not found
        """
        return self._resources[name]

    def __contains__(self, name: str) -> bool:
        """Check if a resource exists in the context."""
        return name in self._resources
