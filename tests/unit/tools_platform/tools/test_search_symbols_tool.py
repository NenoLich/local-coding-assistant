"""Unit tests for search_symbols tool."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from local_coding_assistant.tools_platform.tools.search_symbols import (
    SearchSymbolsInput,
    SearchSymbolsOutput,
    SymbolType,
    search_symbols_impl,
)


@pytest.fixture
def mock_backend() -> MagicMock:
    """Create a mock backend for testing."""
    backend = MagicMock()
    backend.query_repository = AsyncMock()
    return backend


@pytest.fixture
def sample_symbols() -> list[dict]:
    """Sample symbol results."""
    return [
        {
            "name": "hello",
            "symbol_type": "function",
            "file_path": "test.py",
            "line": 1,
            "column": 0,
            "docstring": "Prints hello",
            "signature": "hello()",
        },
        {
            "name": "goodbye",
            "symbol_type": "function",
            "file_path": "test.py",
            "line": 5,
            "column": 0,
            "docstring": "Prints goodbye",
            "signature": "goodbye()",
        },
    ]


@pytest.mark.asyncio
async def test_search_symbols_basic(
    mock_backend: MagicMock, sample_symbols: list[dict]
) -> None:
    """Test basic symbol search."""
    mock_backend.query_repository.return_value = sample_symbols

    input_model = SearchSymbolsInput(
        query="hello",
    )

    result = await search_symbols_impl(mock_backend, input_model)

    assert isinstance(result, SearchSymbolsOutput)
    assert len(result.results) > 0
    assert result.index_available
    assert not result.truncated


@pytest.mark.asyncio
async def test_search_symbols_with_filter(
    mock_backend: MagicMock, sample_symbols: list[dict]
) -> None:
    """Test symbol search with type filter."""
    mock_backend.query_repository.return_value = sample_symbols

    input_model = SearchSymbolsInput(
        query="hello",
        symbol_types=[SymbolType.FUNCTION],
    )

    result = await search_symbols_impl(mock_backend, input_model)

    assert isinstance(result, SearchSymbolsOutput)
    assert len(result.results) > 0
    # Verify backend was called with correct params
    mock_backend.query_repository.assert_called_once_with(
        "search_symbols",
        {
            "query": "hello",
            "symbol_types": ["function"],
            "file_paths": None,
            "limit": 20,
            "offset": 0,
        },
    )


@pytest.mark.asyncio
async def test_search_symbols_with_file_filter(
    mock_backend: MagicMock, sample_symbols: list[dict]
) -> None:
    """Test symbol search with file path filter."""
    mock_backend.query_repository.return_value = sample_symbols

    input_model = SearchSymbolsInput(
        query="hello",
        file_paths=["test.py"],
    )

    result = await search_symbols_impl(mock_backend, input_model)

    assert isinstance(result, SearchSymbolsOutput)
    mock_backend.query_repository.assert_called_once_with(
        "search_symbols",
        {
            "query": "hello",
            "symbol_types": None,
            "file_paths": ["test.py"],
            "limit": 20,
            "offset": 0,
        },
    )


@pytest.mark.asyncio
async def test_search_symbols_limit(
    mock_backend: MagicMock, sample_symbols: list[dict]
) -> None:
    """Test symbol search with limit."""
    mock_backend.query_repository.return_value = sample_symbols

    input_model = SearchSymbolsInput(
        query="hello",
        limit=5,
    )

    result = await search_symbols_impl(mock_backend, input_model)

    assert isinstance(result, SearchSymbolsOutput)
    mock_backend.query_repository.assert_called_once_with(
        "search_symbols",
        {
            "query": "hello",
            "symbol_types": None,
            "file_paths": None,
            "limit": 5,
            "offset": 0,
        },
    )


@pytest.mark.asyncio
async def test_search_symbols_index_unavailable(mock_backend: MagicMock) -> None:
    """Test symbol search when index is unavailable."""

    # Make the mock raise ToolExecutionError
    async def raise_error(*args, **kwargs):
        raise RuntimeError("Index not available")

    mock_backend.query_repository = raise_error

    input_model = SearchSymbolsInput(
        query="hello",
    )

    # Since it's not a ToolExecutionError, it should raise
    with pytest.raises(RuntimeError):
        await search_symbols_impl(mock_backend, input_model)


@pytest.mark.asyncio
async def test_search_symbols_no_results(mock_backend: MagicMock) -> None:
    """Test symbol search with no results."""
    mock_backend.query_repository.return_value = []

    input_model = SearchSymbolsInput(
        query="nonexistent",
    )

    result = await search_symbols_impl(mock_backend, input_model)

    assert isinstance(result, SearchSymbolsOutput)
    assert len(result.results) == 0
    assert result.total_results == 0
