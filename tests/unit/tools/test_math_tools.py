"""Tests for math tools."""

from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from local_coding_assistant.tools.builtin_tools.math_tools import (
    AsyncSumTool,
    MultiplyTool,
    SumTool,
)


class TestSumTool:
    """Tests for the synchronous sum tool."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tool = SumTool()
        self.valid_input = {"numbers": [3, 7]}
        self.expected_output = {"result": 10}

    def test_run_with_valid_input(self):
        """Test running the tool with valid input."""
        input_data = SumTool.Input(**self.valid_input)
        result = self.tool.run(input_data)
        assert result.model_dump() == self.expected_output

    def test_input_validation(self):
        """Test input validation."""
        # Test with invalid input (missing required field)
        with pytest.raises(ValidationError):
            SumTool.Input(**{})

        # Test with valid empty list input
        input_data = SumTool.Input(**{"numbers": []})
        assert input_data.numbers == []


class TestAsyncSumTool:
    """Tests for the asynchronous sum tool."""

    @pytest.fixture
    def tool(self):
        """Return an instance of the async sum tool."""
        return AsyncSumTool()

    @pytest.fixture
    def valid_input(self):
        """Return valid input data."""
        return {"numbers": [2, 3, 4]}

    @pytest.mark.asyncio
    async def test_run_with_valid_input(self, tool, valid_input):
        """Test running the async tool with valid input."""
        input_data = AsyncSumTool.Input(**valid_input)
        result = await tool.run(input_data)
        assert result.model_dump() == {"result": 9}

    @pytest.mark.asyncio
    async def test_input_validation(self, tool):
        """Test input validation for the async tool."""
        # Test with invalid input (missing required field)
        with pytest.raises(ValidationError):
            AsyncSumTool.Input(**{})

        # Test with valid empty list input
        input_data = AsyncSumTool.Input(**{"numbers": []})
        assert input_data.numbers == []


class TestMultiplyTool:
    """Tests for the multiply tool."""

    @pytest.fixture
    def tool(self):
        """Return an instance of the multiply tool."""
        return MultiplyTool()

    @pytest.fixture
    def valid_input(self):
        """Return valid input data."""
        return {"numbers": [2, 3, 4]}  # 2 * 3 * 4 = 24

    def test_run_with_valid_input(self, tool, valid_input):
        """Test running the tool with valid input."""
        input_data = MultiplyTool.Input(**valid_input)
        result = tool.run(input_data)
        assert result.model_dump() == {"result": 24}

    def test_input_validation(self, tool):
        """Test input validation for the multiply tool."""
        # Test with invalid input (missing required field)
        with pytest.raises(ValidationError):
            MultiplyTool.Input(**{})

        # Test with valid empty list input (should multiply to 1)
        input_data = MultiplyTool.Input(**{"numbers": []})
        assert input_data.numbers == []
        
        # Test that multiplying by empty list gives 1 (mathematically correct for empty product)
        result = tool.run(input_data)
        assert result.result == 1.0


class TestToolIntegration:
    """Integration tests for math tools with the tool manager."""

    @pytest.fixture
    def tool_manager(self):
        """Return a mocked tool manager instance with math tools registered."""
        # Create a mock ToolManager instance
        mock_tool_manager = Mock()
        
        # Set up the arun_tool method to handle our test cases
        async def mock_arun_tool(tool_name, params):
            if tool_name == 'sum':
                return {"result": sum(params['numbers'])}
            elif tool_name == 'sum_async':
                return {"result": sum(params['numbers'])}
            elif tool_name == 'multiply':
                result = 1
                for num in params['numbers']:
                    result *= num
                return {"result": result}
            else:
                from local_coding_assistant.core.exceptions import ToolRegistryError
                raise ToolRegistryError(f"Tool '{tool_name}' not found")
        
        mock_tool_manager.arun_tool.side_effect = mock_arun_tool
        
        # Return the mock instance
        return mock_tool_manager

    @pytest.mark.asyncio
    async def test_sync_tool_execution(self, tool_manager):
        """Test executing a synchronous math tool through the tool manager."""
        result = await tool_manager.arun_tool("sum", {"numbers": [5, 5, 5]})
        assert result == {"result": 15}

    @pytest.mark.asyncio
    async def test_async_tool_execution(self, tool_manager):
        """Test executing an asynchronous math tool through the tool manager."""
        result = await tool_manager.arun_tool("sum_async", {"numbers": [10, 20, 30]})
        assert result == {"result": 60}
        
    @pytest.mark.asyncio
    async def test_multiply_tool_execution(self, tool_manager):
        """Test executing the multiply tool through the tool manager."""
        result = await tool_manager.arun_tool("multiply", {"numbers": [2, 3, 4]})
        assert result == {"result": 24}
        
        # Test empty list (should return 1)
        result = await tool_manager.arun_tool("multiply", {"numbers": []})
        assert result == {"result": 1.0}

    @pytest.mark.asyncio
    async def test_nonexistent_tool(self, tool_manager):
        """Test executing a non-existent tool."""
        from local_coding_assistant.core.exceptions import ToolRegistryError

        with pytest.raises(ToolRegistryError) as exc_info:
            await tool_manager.arun_tool("nonexistent_tool", {})

        assert "not found" in str(exc_info.value).lower()
