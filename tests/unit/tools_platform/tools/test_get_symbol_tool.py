"""Unit tests for get_symbol tool."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from local_coding_assistant.tools_platform.tools.get_symbol import (
    GetSymbolInput,
    GetSymbolOutput,
    get_symbol_impl,
)


@pytest.fixture
def mock_backend() -> MagicMock:
    """Create a mock backend for testing."""
    backend = MagicMock()
    backend.query_repository = AsyncMock()
    return backend


@pytest.fixture
def sample_symbol() -> dict:
    """Sample symbol detail."""
    return {
        "name": "hello",
        "symbol_type": "function",
        "file_path": "test.py",
        "line": 1,
        "column": 0,
        "docstring": "Prints hello",
        "signature": "hello()",
    }


@pytest.mark.asyncio
async def test_get_symbol_by_name(mock_backend: MagicMock, sample_symbol: dict) -> None:
    """Test getting symbol by name."""
    mock_backend.query_repository.return_value = [sample_symbol]

    input_model = GetSymbolInput(
        name="hello",
    )

    result = await get_symbol_impl(mock_backend, input_model)

    assert isinstance(result, GetSymbolOutput)
    assert result.found
    assert len(result.symbols) == 1
    assert result.symbols[0].name == "hello"
    assert result.index_available


@pytest.mark.asyncio
async def test_get_symbol_with_path(
    mock_backend: MagicMock, sample_symbol: dict
) -> None:
    """Test getting symbol with file path."""
    mock_backend.query_repository.return_value = [sample_symbol]

    input_model = GetSymbolInput(
        name="hello",
        path="test.py",
    )

    result = await get_symbol_impl(mock_backend, input_model)

    assert isinstance(result, GetSymbolOutput)
    assert result.found
    mock_backend.query_repository.assert_called_once_with(
        "get_symbol",
        {
            "name": "hello",
            "path": "test.py",
            "symbol_type": None,
            "limit": 20,
        },
    )


@pytest.mark.asyncio
async def test_get_symbol_with_type_filter(
    mock_backend: MagicMock, sample_symbol: dict
) -> None:
    """Test getting symbol with type filter."""
    mock_backend.query_repository.return_value = [sample_symbol]

    input_model = GetSymbolInput(
        name="hello",
        symbol_type="function",
    )

    result = await get_symbol_impl(mock_backend, input_model)

    assert isinstance(result, GetSymbolOutput)
    assert result.found
    mock_backend.query_repository.assert_called_once_with(
        "get_symbol",
        {
            "name": "hello",
            "path": None,
            "symbol_type": "function",
            "limit": 20,
        },
    )


@pytest.mark.asyncio
async def test_get_symbol_not_found(mock_backend: MagicMock) -> None:
    """Test getting symbol that doesn't exist."""
    mock_backend.query_repository.return_value = None

    input_model = GetSymbolInput(
        name="nonexistent",
    )

    result = await get_symbol_impl(mock_backend, input_model)

    assert isinstance(result, GetSymbolOutput)
    assert not result.found
    assert result.symbols == []
    assert result.index_available


@pytest.mark.asyncio
async def test_get_symbol_multiple_returns(mock_backend: MagicMock) -> None:
    """Test getting multiple symbols sharing a name."""
    sample_symbols = [
        {
            "name": "foo",
            "symbol_type": "method",
            "file_path": "a.py",
            "line": 1,
            "column": 0,
            "docstring": "method on A",
            "signature": "foo(self)",
        },
        {
            "name": "foo",
            "symbol_type": "method",
            "file_path": "b.py",
            "line": 5,
            "column": 0,
            "docstring": "method on B",
            "signature": "foo(self)",
        },
    ]
    mock_backend.query_repository.return_value = sample_symbols

    input_model = GetSymbolInput(
        name="foo",
        limit=10,
    )

    result = await get_symbol_impl(mock_backend, input_model)

    assert isinstance(result, GetSymbolOutput)
    assert result.found
    assert len(result.symbols) == 2
    assert result.symbols[0].name == "foo"
    assert result.symbols[1].file_path == "b.py"


@pytest.mark.asyncio
async def test_get_symbol_index_unavailable(mock_backend: MagicMock) -> None:
    """Test getting symbol when index is unavailable."""

    # Make the mock raise a non-ToolExecutionError exception
    async def raise_error(*args, **kwargs):
        raise RuntimeError("Index not available")

    mock_backend.query_repository = raise_error

    input_model = GetSymbolInput(
        name="hello",
    )

    # Since it's not a ToolExecutionError, it should raise
    with pytest.raises(RuntimeError):
        await get_symbol_impl(mock_backend, input_model)
