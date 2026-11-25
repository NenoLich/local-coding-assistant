"""Get symbol tool implementation.

Exact symbol detail by path/name with bounded source range.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from local_coding_assistant.tools_platform.contracts import (
    BackendType,
    ConcurrencyScope,
    LockMode,
    OperationEffect,
    ToolRiskClass,
)


class GetSymbolInput(BaseModel):
    """Input model for get_symbol tool."""

    name: str = Field(description="Symbol name to search for")
    path: str | None = Field(default=None, description="File path to search in")
    symbol_type: str | None = Field(default=None, description="Symbol type filter")
    limit: int = Field(default=20, description="Maximum number of symbols to return")


class SymbolDetail(BaseModel):
    """Detailed symbol information."""

    name: str = Field(description="Symbol name")
    symbol_type: str = Field(description="Symbol type (function, class, method, etc.)")
    file_path: str = Field(description="File path relative to workspace")
    line: int = Field(description="Line number")
    column: int = Field(description="Column number")
    docstring: str | None = Field(default=None, description="Symbol docstring")
    signature: str | None = Field(default=None, description="Function signature")


class GetSymbolOutput(BaseModel):
    """Output model for get_symbol tool."""

    symbols: list[SymbolDetail] = Field(
        default_factory=list, description="Matching symbols (may be multiple)"
    )
    found: bool = Field(description="Whether at least one symbol was found")
    index_available: bool = Field(description="Whether repository index is available")

    def to_llm_dict(self) -> dict[str, Any]:
        """Return LLM-friendly dict."""
        return {
            "symbols": [s.model_dump() for s in self.symbols],
            "found": self.found,
            "index_available": self.index_available,
        }

    @classmethod
    def model_llm_schema(cls) -> dict[str, Any]:
        """Return LLM-friendly JSON schema."""
        from pydantic import create_model

        llm_output = create_model(
            "GetSymbolOutputLLM",
            symbols=(list[SymbolDetail], ...),
            found=(bool, ...),
            index_available=(bool, ...),
        )
        return llm_output.model_json_schema()


async def get_symbol_impl(
    backend: Any,
    input_model: GetSymbolInput,
    session_id: str | None = None,
    turn_id: str | None = None,
) -> GetSymbolOutput:
    """Implementation of get_symbol tool.

    Args:
        backend: ExecutionBackend instance
        input_model: Validated input parameters
        session_id: Session ID for conflict detection
        turn_id: Turn ID for conflict detection

    Returns:
        GetSymbolOutput with matching symbols (supports multiple returns)
    """
    # Use backend's query_repository method
    params = {
        "name": input_model.name,
        "path": input_model.path,
        "symbol_type": input_model.symbol_type,
        "limit": input_model.limit,
    }

    try:
        result = await backend.query_repository("get_symbol", params)
        index_available = True
    except Exception as e:
        # If index is unavailable, return not found
        from local_coding_assistant.tools_platform.exceptions import ToolPlatformError

        if isinstance(e, ToolPlatformError):
            index_available = False
            result = None
        else:
            raise

    if result:
        symbol_details = [SymbolDetail(**r) for r in result]
        return GetSymbolOutput(
            symbols=symbol_details,
            found=True,
            index_available=index_available,
        )
    else:
        return GetSymbolOutput(
            symbols=[],
            found=False,
            index_available=index_available,
        )


def register_get_symbol(registry: Any) -> None:
    """Register get_symbol tool with the registry.

    Args:
        registry: ToolRegistry instance
    """
    from datetime import timedelta

    from local_coding_assistant.tools_platform.contracts import (
        ConflictRule,
        LockLifetime,
        ResourceLock,
        TimeoutProfile,
    )
    from local_coding_assistant.tools_platform.definitions import ToolBuilder

    builder = (
        ToolBuilder(
            tool_id="get_symbol",
            description="Gets exact symbol detail by name with optional path and type filters. Returns symbol metadata and location. Only within the working directory.",
        )
        .with_risk_class(ToolRiskClass.SAFE)
        .with_effects(OperationEffect.READ)
        .with_backends(BackendType.HOST, BackendType.DOCKER)
        .with_concurrency_scope(
            ConcurrencyScope(
                effect=OperationEffect.READ,
                resource_locks=[
                    ResourceLock(key="workspace", mode=LockMode.SHARED),
                    ResourceLock(key="repository_index", mode=LockMode.SHARED),
                ],
                conflict_rule=ConflictRule.WAIT,
                lifetime=LockLifetime.EXECUTION,
            )
        )
        .with_timeout_profile(
            TimeoutProfile(
                soft_timeout=timedelta(seconds=5),
                hard_timeout=timedelta(seconds=30),
                idle_timeout=timedelta(seconds=30),
            )
        )
        .with_implementation(
            factory=get_symbol_impl,
            input_model=GetSymbolInput,
            output_model=GetSymbolOutput,
        )
    )

    definition, implementation = builder.build_with_implementation()
    registry.register(definition, implementation)
