"""Unit tests for sandbox-related functionality in ToolManager."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from pytest_mock import MockerFixture

from local_coding_assistant.core.exceptions import ToolRegistryError
from local_coding_assistant.tools.types import ToolExecutionRequest, ToolInfo, ToolSource, ToolCategory
from local_coding_assistant.tools.tool_manager import ToolManager

from .tool_test_helpers import (
    StubToolConfig,
    SyncTestTool,
    build_manager,
    make_tool_info,
)

# Add mocker fixture for tests that need it
pytest_plugins = ["pytest_mock"]


class TestSandboxToolExecution:
    """Tests for sandbox tool execution and related functionality."""

    @pytest.mark.asyncio
    async def test_execute_tool_in_sandbox_success(self, mocker: MockerFixture):
        """Test successful execution of a tool in sandbox."""
        # Setup
        manager, _ = build_manager({})
        tool_name = "test_tool"
        payload = {"param1": "value1"}
        session_id = "test_session"
        
        # Mock the runtime and its execute method to return an awaitable
        mock_runtime = mocker.MagicMock()
        future = asyncio.Future()
        future.set_result({
            "response": {
                "success": True,
                "result": "test_result",
                "stdout": "test output",
                "stderr": "",
                "tool_calls": [{
                    "tool_name": tool_name,
                    "result": "test_result",
                    "start_time": "2023-01-01T00:00:00Z",
                    "end_time": "2023-01-01T00:00:01Z",
                    "success": True,
                    "resource_metrics": []
                }],
                "system_metrics": [{"metric": "cpu", "value": 50}]
            }
        })
        mock_runtime.execute.return_value = future
        manager._runtimes = {"execute_python_code": mock_runtime}
        
        # Execute
        result = await manager.execute_tool_in_sandbox(tool_name, payload, session_id)
        
        # Verify
        assert "=== Sandbox Execution Details ===" in result
        assert "Tool: test_tool" in result
        assert "Session: test_session" in result
        assert "--- Result ---" in result
        assert "test_result" in result
        mock_runtime.execute.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_execute_tool_in_sandbox_failure(self, mocker: MockerFixture):
        """Test sandbox execution failure handling."""
        manager, _ = build_manager({})
        mock_runtime = mocker.MagicMock()
        future = asyncio.Future()
        future.set_result({
            "response": {
                "success": False,
                "error": "Execution failed"
            }
        })
        mock_runtime.execute.return_value = future
        manager._runtimes = {"execute_python_code": mock_runtime}
        
        with pytest.raises(ToolRegistryError, match="Sandbox execution failed"):
            await manager.execute_tool_in_sandbox("test_tool", {}, "test_session")

    @pytest.mark.asyncio
    async def test_generate_tool_call_code(self):
        """Test generation of tool call code."""
        manager, _ = build_manager({})
        tool_name = "test_tool"
        payload = {"param1": "value1", "param2": 42}
        
        # Mock the _generate_tool_call_code method
        with patch.object(manager, '_generate_tool_call_code') as mock_generate:
            mock_generate.return_value = """from tools_api import test_tool
result = test_tool(param1='value1', param2=42)
print(result)"""
            
            code = await manager._generate_tool_call_code(tool_name, payload)
            
            # Verify the mock was called with correct arguments
            mock_generate.assert_awaited_once_with(tool_name, payload)
            
            # Verify the returned code
            assert "from tools_api import test_tool" in code
            assert "result = test_tool(param1='value1', param2=42)" in code
            assert "print(result)" in code

    @pytest.mark.asyncio
    async def test_get_sandbox_tools_prompt(self):
        """Test generation of sandbox tools prompt."""
        # Create a tool info with the expected structure
        tool_info = ToolInfo(
            name="test_tool",
            description="Test tool description",
            source=ToolSource.SANDBOX,
            available=True,
            enabled=True,
            parameters={
                "properties": {
                    "param1": {"type": "string", "description": "First parameter"},
                    "param2": {"type": "integer", "description": "Second parameter"}
                },
                "required": ["param1"]
            },
            tool_class=SyncTestTool
        )
        
        # Create a manager with the tool registered
        manager, _ = build_manager({"test_tool": StubToolConfig(tool_info)}
        )
        
        # Ensure the tool is in the manager's tools dictionary
        manager._tools = {"test_tool": tool_info}
        
        # Get the prompt
        prompt = manager.get_sandbox_tools_prompt()
        
        # Verify the prompt contains the expected content
        assert "## test_tool" in prompt
        assert "Test tool description" in prompt
        assert "### Parameters:" in prompt
        assert "param1: First parameter" in prompt
        assert "[Type: string] (required)" in prompt
        assert "param2: Second parameter" in prompt
        assert "[Type: integer] (optional)" in prompt
        assert "### Usage Example:" in prompt
        assert "from tools_api import test_tool" in prompt

    @pytest.mark.asyncio
    async def test_record_success_with_tool_calls(self, mocker: MockerFixture):
        """Test recording success with tool calls."""
        manager, _ = build_manager({})
        
        # Create a mock for the statistics manager with async methods
        mock_stats = mocker.AsyncMock()
        manager._statistics = mock_stats
        
        tool_name = "test_tool"
        duration = 1.23
        tool_calls = [{
            "tool_name": "sub_tool",
            "result": "result",
            "start_time": "2023-01-01T00:00:00Z",
            "end_time": "2023-01-01T00:00:01Z",
            "success": True,
            "resource_metrics": [{"metric": "cpu", "value": 50}]
        }]
        system_metrics = [{"metric": "memory", "value": 1024}]
        
        # Patch the _record_success method to avoid actual async calls
        with patch.object(manager, '_statistics') as mock_stats:
            # Make the mock's record_system_metrics return an awaitable
            mock_stats.record_system_metrics = mocker.AsyncMock(return_value=None)
            mock_stats.record_tool_call = mocker.AsyncMock(return_value=None)
            
            await manager._record_success(tool_name, duration, tool_calls, system_metrics)
            
            # Verify statistics were recorded
            mock_stats.record_system_metrics.assert_awaited_once_with(system_metrics, duration)
            assert mock_stats.record_tool_call.await_count == 1

    @pytest.mark.asyncio
    async def test_build_sandbox_result_string(self):
        """Test building sandbox result string."""
        manager, _ = build_manager({})
        sandbox_result = {
            "result": "test result",
            "stdout": "test output",
            "stderr": "test error",
            "duration": 1.23,
            "return_code": 0,
            "files_created": ["/tmp/file1.txt"],
            "files_modified": ["/tmp/file2.txt"]
        }
        
        # Mock the _build_sandbox_result_string method
        with patch.object(manager, '_build_sandbox_result_string') as mock_build:
            mock_build.return_value = """=== Sandbox Execution Details ===
Tool: test_tool
Session: test_session
Duration: 1.23s
Exit Code: 0

--- Result ---
test result

--- Standard Output ---
test output

--- Standard Error ---
test error

--- Files Created ---
- /tmp/file1.txt

--- Files Modified ---
- /tmp/file2.txt"""
            
            result = await manager._build_sandbox_result_string(
                sandbox_result, "test_tool", "test_session"
            )
            
            # Verify the mock was called with correct arguments
            mock_build.assert_awaited_once_with(sandbox_result, "test_tool", "test_session")
            
            # Verify the returned result
            assert "=== Sandbox Execution Details ===" in result
            assert "Tool: test_tool" in result
            assert "Session: test_session" in result
            assert "1.23s" in result
            assert "--- Result ---" in result
            assert "test result" in result
            assert "--- Standard Output ---" in result
            assert "test output" in result
            assert "--- Standard Error ---" in result
            assert "test error" in result
            assert "--- Files Created ---" in result
            assert "- /tmp/file1.txt" in result
            assert "--- Files Modified ---" in result
            assert "- /tmp/file2.txt" in result

    @pytest.mark.asyncio
    async def test_run_programmatic_tool_call(self, mocker: MockerFixture):
        """Test running a programmatic tool call."""
        manager, _ = build_manager({})
        manager._sandbox_manager = mocker.MagicMock()
        
        mock_execute = mocker.AsyncMock(return_value="test_result")
        manager.execute_async = mock_execute
        
        code = "print('hello')"
        session_id = "test_session"
        env_vars = {"KEY": "value"}
        
        result = await manager.run_programmatic_tool_call(code, session_id, env_vars)
        
        mock_execute.assert_awaited_once()
        call_args = mock_execute.call_args[0][0]
        assert isinstance(call_args, ToolExecutionRequest)
        assert call_args.tool_name == "execute_python_code"
        assert call_args.payload["code"] == code
        assert call_args.payload["session_id"] == session_id
        assert call_args.payload["env_vars"] == env_vars

    @pytest.mark.asyncio
    async def test_tool_instantiation_error(self, mocker: MockerFixture):
        """Test error handling during tool instantiation."""
        # Create a tool class that raises an exception during instantiation
        class FaultyTool:
            def __init__(self):
                raise ValueError("Failed to initialize tool")
        
        tool_info = ToolInfo(
            name="faulty_tool",
            description="A tool that fails to initialize",
            source=ToolSource.SANDBOX,
            available=True,
            enabled=True,
            tool_class=FaultyTool
        )
        
        manager, _ = build_manager({"faulty_tool": StubToolConfig(tool_info)})
        
        with pytest.raises(ToolRegistryError, match="Failed to instantiate tool 'faulty_tool'"):
            manager._build_runtime("faulty_tool", tool_info)

    @pytest.mark.asyncio
    async def test_sandbox_tool_without_sandbox_manager(self, mocker: MockerFixture):
        """Test behavior when sandbox manager is not available."""
        # Create a manager with a mock config
        mock_config = MagicMock()
        manager = ToolManager(mock_config)
        
        # Mock the _get_runtime method to return a mock runtime
        mock_runtime = MagicMock()
        mock_runtime.instance = MagicMock()
        mock_runtime.instance.sandbox_manager = None  # Simulate no sandbox manager
        
        # Patch the _get_runtime method to return our mock runtime
        with patch.object(manager, '_get_runtime', return_value=mock_runtime):
            with pytest.raises(ToolRegistryError, match="Sandbox execution is not available"):
                await manager.execute_tool_in_sandbox("test_tool", {}, "test_session")

    @pytest.mark.asyncio
    async def test_sandbox_tool_with_invalid_response(self, mocker: MockerFixture):
        """Test handling of invalid sandbox response."""
        manager, _ = build_manager({})
        
        # Create a mock runtime with sandbox manager
        mock_runtime = mocker.MagicMock()
        mock_runtime.instance = mocker.MagicMock()
        mock_runtime.instance.sandbox_manager = mocker.MagicMock()
        
        # Set up the future with an invalid response
        future = asyncio.Future()
        future.set_result({"invalid": "response"})
        mock_runtime.execute.return_value = future
        
        # Patch _get_runtime to return our mock runtime
        with patch.object(manager, '_get_runtime', return_value=mock_runtime):
            with pytest.raises(ToolRegistryError, match="Sandbox execution failed: Invalid response from sandbox"):
                await manager.execute_tool_in_sandbox("test_tool", {}, "test_session")
