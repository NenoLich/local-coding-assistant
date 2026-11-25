"""Search symbols tool implementation.

AST/database-backed symbol lookup with name/query, symbol-type,
file/path filters, pagination, and repository-index status.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from local_coding_assistant.tools_platform.contracts import (
    BackendType,
    ConcurrencyScope,
    LockMode,
    OperationEffect,
    ToolRiskClass,
)


class SymbolType(str, Enum):
    """Symbol type filter."""

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    VARIABLE = "variable"
    CONSTANT = "constant"
    INTERFACE = "interface"
    ENUM = "enum"
    TYPE_ALIAS = "type_alias"
    MODULE = "module"


class SearchSymbolsInput(BaseModel):
    """Input model for search_symbols tool."""

    query: str = Field(description="Search query for symbol names")
    symbol_types: list[SymbolType] | None = Field(
        default=None,
        description="Filter by symbol types",
    )
    file_paths: list[str] | None = Field(
        default=None, description="Filter by specific file paths"
    )
    limit: int = Field(default=20, description="Maximum number of results to return")
    offset: int = Field(default=0, description="Number of results to skip (pagination)")


class SymbolResult(BaseModel):
    """A single symbol search result."""

    name: str = Field(description="Symbol name")
    symbol_type: str = Field(description="Symbol type (function, class, method, etc.)")
    file_path: str = Field(description="File path relative to workspace")
    line: int = Field(description="Line number")
    column: int = Field(description="Column number")
    docstring: str | None = Field(default=None, description="Symbol docstring")
    signature: str | None = Field(default=None, description="Function signature")


class SearchSymbolsOutput(BaseModel):
    """Output model for search_symbols tool."""

    results: list[SymbolResult] = Field(description="List of matching symbols")
    total_results: int = Field(description="Total number of results found")
    index_available: bool = Field(description="Whether repository index is available")
    truncated: bool = Field(description="Whether results were truncated")
    next_offset: int | None = Field(
        description="Next offset for pagination if truncated"
    )

    def to_llm_dict(self) -> dict[str, Any]:
        """Return LLM-friendly dict."""
        return {
            "results": [r.model_dump() for r in self.results],
            "total_results": self.total_results,
            "index_available": self.index_available,
            "truncated": self.truncated,
            "next_offset": self.next_offset,
        }

    @classmethod
    def model_llm_schema(cls) -> dict[str, Any]:
        """Return LLM-friendly JSON schema."""
        from pydantic import create_model

        llm_output = create_model(
            "SearchSymbolsOutputLLM",
            results=(list[SymbolResult], ...),
            total_results=(int, ...),
            index_available=(bool, ...),
            truncated=(bool, ...),
            next_offset=(int | None, None),
        )
        return llm_output.model_json_schema()


async def search_symbols_impl(
    backend: Any,
    input_model: SearchSymbolsInput,
    session_id: str | None = None,
    turn_id: str | None = None,
) -> SearchSymbolsOutput:
    """Implementation of search_symbols tool.

    Args:
        backend: ExecutionBackend instance
        input_model: Validated input parameters
        session_id: Session ID for conflict detection
        turn_id: Turn ID for conflict detection

    Returns:
        SearchSymbolsOutput with symbol results
    """
    # Use backend's query_repository method
    params = {
        "query": input_model.query,
        "symbol_types": [st.value for st in input_model.symbol_types]
        if input_model.symbol_types
        else None,
        "file_paths": input_model.file_paths,
        "limit": input_model.limit,
        "offset": input_model.offset,
    }

    try:
        results = await backend.query_repository("search_symbols", params)
        index_available = True
    except Exception as e:
        # If index is unavailable, return empty results
        from local_coding_assistant.tools_platform.exceptions import ToolPlatformError

        if isinstance(e, ToolPlatformError):
            index_available = False
            return SearchSymbolsOutput(
                results=[],
                total_results=0,
                index_available=False,
                truncated=False,
                next_offset=None,
            )
        else:
            raise

    # Convert results to SymbolResult models
    symbol_results = [SymbolResult(**r) for r in results] if results else []

    truncated = len(symbol_results) >= input_model.limit
    next_offset = input_model.offset + len(symbol_results) if truncated else None

    return SearchSymbolsOutput(
        results=symbol_results,
        total_results=len(symbol_results),
        index_available=index_available,
        truncated=truncated,
        next_offset=next_offset,
    )


def register_search_symbols(registry: Any) -> None:
    """Register search_symbols tool with the registry.

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
            tool_id="search_symbols",
            description="Searches for symbols (functions, classes, methods) using AST/database index. Supports filtering by symbol type and file paths. Returns symbol details with file locations. Supports pagination. Only within the working directory.",
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
            factory=search_symbols_impl,
            input_model=SearchSymbolsInput,
            output_model=SearchSymbolsOutput,
        )
    )

    definition, implementation = builder.build_with_implementation()
    registry.register(definition, implementation)
