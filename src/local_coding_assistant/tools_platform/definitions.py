"""Tool definition and registration infrastructure.

This module provides the infrastructure for defining and registering tools
with complete metadata including input/output models, permissions, effects,
and backend capabilities.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel

from local_coding_assistant.tools_platform.contracts import (
    BackendType,
    ConcurrencyScope,
    OperationEffect,
    TimeoutProfile,
    ToolDefinition,
    ToolRiskClass,
)

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class ToolImplementation:
    """Tool implementation factory and metadata.

    Attributes:
        factory: Factory function that creates the tool implementation
        input_model: Pydantic model for input validation
        output_model: Pydantic model for output validation
    """

    factory: Callable[..., Any]
    input_model: type[BaseModel]
    output_model: type[BaseModel] | None = None


class ToolBuilder:
    """Builder for constructing ToolDefinition instances.

    Provides a fluent interface for building tool definitions with
    all required metadata.
    """

    def __init__(
        self,
        tool_id: str,
        description: str,
    ) -> None:
        """Initialize the builder.

        Args:
            tool_id: Stable tool identifier
            description: Human-readable description for LLM
        """
        self._id = tool_id
        self._description = description
        self._version = "1.0.0"
        self._permissions: list[str] = []
        self._risk_class = ToolRiskClass.LOW
        self._effects: list[OperationEffect] = []
        self._supported_backends: list[BackendType] = [BackendType.HOST]
        self._supports_streaming = False
        self._concurrency_scope: ConcurrencyScope | None = None
        self._timeout_profile: TimeoutProfile | None = None
        self._exposure_policy: dict[str, Any] = {}
        self._implementation: ToolImplementation | None = None

    def with_version(self, version: str) -> ToolBuilder:
        """Set the tool version.

        Args:
            version: Tool version string

        Returns:
            Self for chaining
        """
        self._version = version
        return self

    def with_permissions(self, *permissions: str) -> ToolBuilder:
        """Set required permissions.

        Args:
            *permissions: Permission strings

        Returns:
            Self for chaining
        """
        self._permissions = list(permissions)
        return self

    def with_risk_class(self, risk_class: ToolRiskClass) -> ToolBuilder:
        """Set the risk classification.

        Args:
            risk_class: Risk class

        Returns:
            Self for chaining
        """
        self._risk_class = risk_class
        return self

    def with_effects(self, *effects: OperationEffect) -> ToolBuilder:
        """Set operation effects.

        Args:
            *effects: Operation effects

        Returns:
            Self for chaining
        """
        self._effects = list(effects)
        return self

    def with_backends(self, *backends: BackendType) -> ToolBuilder:
        """Set supported backends.

        Args:
            *backends: Backend types

        Returns:
            Self for chaining
        """
        self._supported_backends = list(backends)
        return self

    def with_streaming(self, supports: bool = True) -> ToolBuilder:
        """Set streaming support.

        Args:
            supports: Whether tool supports streaming

        Returns:
            Self for chaining
        """
        self._supports_streaming = supports
        return self

    def with_concurrency_scope(self, scope: ConcurrencyScope) -> ToolBuilder:
        """Set concurrency scope.

        Args:
            scope: Concurrency scope

        Returns:
            Self for chaining
        """
        self._concurrency_scope = scope
        return self

    def with_timeout_profile(self, profile: TimeoutProfile) -> ToolBuilder:
        """Set timeout profile.

        Args:
            profile: Timeout profile

        Returns:
            Self for chaining
        """
        self._timeout_profile = profile
        return self

    def with_exposure_policy(self, policy: dict[str, Any]) -> ToolBuilder:
        """Set exposure policy.

        Args:
            policy: Exposure policy dictionary

        Returns:
            Self for chaining
        """
        self._exposure_policy = policy
        return self

    def with_implementation(
        self,
        factory: Callable[..., Any],
        input_model: type[BaseModel],
        output_model: type[BaseModel] | None = None,
    ) -> ToolBuilder:
        """Set the tool implementation.

        Args:
            factory: Factory function for the tool
            input_model: Pydantic input model
            output_model: Optional Pydantic output model

        Returns:
            Self for chaining
        """
        self._implementation = ToolImplementation(
            factory=factory,
            input_model=input_model,
            output_model=output_model,
        )
        return self

    def build(self) -> ToolDefinition:
        """Build the ToolDefinition.

        Returns:
            Complete ToolDefinition

        Raises:
            ValueError: If required fields are missing
        """
        if self._implementation is None:
            raise ValueError(f"Tool '{self._id}' must have an implementation")

        input_model = self._implementation.input_model
        output_model = self._implementation.output_model or BaseModel

        return ToolDefinition(
            id=self._id,
            version=self._version,
            description=self._description,
            input_schema=input_model.model_json_schema(),
            output_schema=self._get_llm_output_schema(output_model),
            permissions=self._permissions,
            risk_class=self._risk_class,
            effects=self._effects,
            supported_backends=self._supported_backends,
            supports_streaming=self._supports_streaming,
            concurrency_scope=self._concurrency_scope,
            timeout_profile=self._timeout_profile,
            exposure_policy=self._exposure_policy,
        )

    def _get_llm_output_schema(self, output_model: type[BaseModel]) -> dict[str, Any]:
        """Get LLM-friendly output schema (excludes internal fields)."""
        if hasattr(output_model, "model_llm_schema"):
            return output_model.model_llm_schema()
        return output_model.model_json_schema()

    def build_with_implementation(self) -> tuple[ToolDefinition, ToolImplementation]:
        """Build both ToolDefinition and ToolImplementation.

        Returns:
            Tuple of (ToolDefinition, ToolImplementation)

        Raises:
            ValueError: If required fields are missing
        """
        if self._implementation is None:
            raise ValueError(f"Tool '{self._id}' must have an implementation")

        definition = self.build()
        return definition, self._implementation
