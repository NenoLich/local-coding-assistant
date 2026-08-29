"""Repository tools for code analysis and symbol search."""

from pydantic import BaseModel, Field

from local_coding_assistant.tools.tool_executor import get_tool_executor
from local_coding_assistant.tools.tool_registry import register_tool
from local_coding_assistant.tools.types import ToolCategory, ToolTag


@register_tool(
    name="search_symbols",
    description="Search for symbols (functions, classes, methods) across the codebase",
    category=ToolCategory.SEARCH,
    tags=[ToolTag.CODE, ToolTag.FILESYSTEM, ToolTag.RETRIEVAL],
    permissions=[],
    is_async=False,
    supports_streaming=False,
)
class SearchSymbolsTool:
    """A tool that searches for symbols across the codebase."""

    class Input(BaseModel):
        """Input model for the symbol search tool."""

        query: str = Field(..., description="Search query for symbol names")
        symbol_types: list[str] | None = Field(
            None, description="Filter by symbol types (function, class, method, etc.)"
        )
        file_path: str | None = Field(None, description="Filter by specific file path")
        limit: int = Field(20, description="Maximum number of results")

    class Output(BaseModel):
        """Output model for the symbol search tool."""

        results: list[dict] = Field(..., description="List of matching symbols")
        available: bool = Field(
            ..., description="Whether repository service is available"
        )

    def run(self, input_data: Input) -> Output:
        """Search for symbols across the codebase.

        Args:
            input_data: Input containing search parameters

        Returns:
            The search results

        Raises:
            RuntimeError: If tool executor is not initialized
        """
        try:
            tool_executor = get_tool_executor()
            results = tool_executor.search_symbols(
                query=input_data.query,
                symbol_types=input_data.symbol_types,
                file_path=input_data.file_path,
                limit=input_data.limit,
            )
            return self.Output(results=results, available=True)
        except RuntimeError as e:
            return self.Output(results=[{"error": str(e)}], available=False)
