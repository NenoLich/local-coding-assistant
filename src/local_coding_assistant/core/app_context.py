"""Application context holding shared resources and state."""

from typing import Any, TypeVar

from local_coding_assistant.core.dependencies import AppDependencies

T = TypeVar("T")


class AppContext:
    """Central context object that holds application-wide resources and dependencies."""

    def __init__(self, dependencies: AppDependencies | None = None) -> None:
        """Initialize the application context.

        Args:
            dependencies: Optional pre-initialized dependencies
        """
        self._resources: dict[str, Any] = {}
        self._dependencies = dependencies

    @property
    def deps(self) -> AppDependencies:
        """Get the application dependencies.

        Returns:
            The AppDependencies instance

        Raises:
            RuntimeError: If dependencies are not initialized
        """
        if self._dependencies is None:
            raise RuntimeError(
                "Dependencies not initialized. Did you call bootstrap()?"
            )
        return self._dependencies

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

    def get_typed(
        self, name: str, expected_type: type[T], default: T | None = None
    ) -> T | None:
        """Retrieve a resource by name with type checking.

        Args:
            name: Name of the resource to retrieve
            expected_type: The expected type of the resource
            default: Default value if resource not found or has wrong type

        Returns:
            The requested resource with the correct type, or default if not found/wrong type
        """
        resource = self._resources.get(name)
        if not isinstance(resource, expected_type):
            return default
        return resource

    def __getitem__(self, name: str) -> Any:
        """Enable dictionary-style access to resources.

        Raises:
            KeyError: If the resource is not found
        """
        return self._resources[name]

    def __contains__(self, name: str) -> bool:
        """Check if a resource exists in the context."""
        return name in self._resources
