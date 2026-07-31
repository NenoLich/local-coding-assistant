"""Tests for tool call processing with proper argument handling."""

from unittest.mock import MagicMock

from local_coding_assistant.core.telemetry_types import ResourceType
from local_coding_assistant.sandbox.docker_sandbox import DockerSandbox
from local_coding_assistant.sandbox.sandbox_types import (
    SandboxExecutionResponse,
    ToolCallMetric,
)


class TestToolCallProcessing:
    """Test tool call processing and argument handling."""

    def test_build_tool_call_input_with_kwargs(self):
        """Test _build_tool_call_input with proper kwargs."""
        call_data = {
            "tool_name": "test_tool",
            "args": (),
            "kwargs": {"operation": "add", "numbers": [1, 2, 3]},
        }

        result = DockerSandbox._build_tool_call_input(call_data)

        assert result == {"operation": "add", "numbers": [1, 2, 3]}

    def test_build_tool_call_input_with_args_only(self):
        """Test _build_tool_call_input with only positional args."""
        call_data = {"tool_name": "test_tool", "args": ("arg1", "arg2"), "kwargs": {}}

        result = DockerSandbox._build_tool_call_input(call_data)

        assert result == {"args": ["arg1", "arg2"]}

    def test_build_tool_call_input_with_single_arg(self):
        """Test _build_tool_call_input with single positional arg."""
        call_data = {"tool_name": "test_tool", "args": ("single_arg",), "kwargs": {}}

        result = DockerSandbox._build_tool_call_input(call_data)

        assert result == {"args": ["single_arg"]}

    def test_build_tool_call_input_with_existing_input(self):
        """Test _build_tool_call_input with pre-existing input dict."""
        call_data = {
            "tool_name": "test_tool",
            "input": {"operation": "multiply", "numbers": [4, 5]},
            "args": (),
            "kwargs": {},
        }

        result = DockerSandbox._build_tool_call_input(call_data)

        assert result == {"operation": "multiply", "numbers": [4, 5]}

    def test_build_tool_call_input_empty(self):
        """Test _build_tool_call_input with no args or kwargs."""
        call_data = {"tool_name": "test_tool", "args": (), "kwargs": {}}

        result = DockerSandbox._build_tool_call_input(call_data)

        assert result is None

    def test_process_tool_call_metrics_with_proper_args(self):
        """Test _process_tool_call_metrics with proper argument structure."""
        # Create mock response
        response = MagicMock(spec=SandboxExecutionResponse)
        response.tool_calls = []

        # Create call data with proper arguments
        call_data = {
            "tool_name": "math",
            "call_id": "call-123",
            "duration": 0.05,
            "success": True,
            "args": (),
            "kwargs": {"operation": "add", "numbers": [1, 2]},
            "result": 3,
            "start_time": "2026-01-01T12:00:00+00:00",
            "end_time": "2026-01-01T12:00:00.050+00:00",
            "metadata": {"test": "value"},
        }

        metrics_per_tool_call = {"tool_calls": [call_data]}

        # Process metrics
        DockerSandbox._process_tool_call_metrics(response, metrics_per_tool_call)

        # Verify results
        assert len(response.tool_calls) == 1
        tool_call = response.tool_calls[0]

        assert isinstance(tool_call, ToolCallMetric)
        assert tool_call.tool_name == "math"
        assert tool_call.call_id == "call-123"
        assert tool_call.success is True
        assert tool_call.input == {"operation": "add", "numbers": [1, 2]}
        assert tool_call.output == 3
        assert tool_call.metadata == {"test": "value"}

    def test_process_tool_call_metrics_with_resource_metrics(self):
        """Test _process_tool_call_metrics with resource metrics."""
        response = MagicMock(spec=SandboxExecutionResponse)
        response.tool_calls = []

        # Create call data with resource metrics
        call_data = {
            "tool_name": "math",
            "call_id": "call-456",
            "duration": 0.1,
            "success": True,
            "args": (),
            "kwargs": {"operation": "multiply", "numbers": [3, 4]},
            "result": 12,
            "start_time": "2026-01-01T12:00:00+00:00",
            "end_time": "2026-01-01T12:00:00.100+00:00",
            "delta_stats": {
                "cpu_delta": 10.5,
                "memory_delta_mb": 5.2,
                "read_bytes_per_sec": 1000,
                "write_bytes_per_sec": 500,
            },
        }

        metrics_per_tool_call = {"tool_calls": [call_data]}

        # Process metrics
        DockerSandbox._process_tool_call_metrics(response, metrics_per_tool_call)

        # Verify results
        assert len(response.tool_calls) == 1
        tool_call = response.tool_calls[0]

        assert tool_call.tool_name == "math"
        assert tool_call.input == {"operation": "multiply", "numbers": [3, 4]}
        assert tool_call.output == 12

        # Check resource metrics were extracted
        assert len(tool_call.resource_metrics) > 0

        # Find CPU metric
        cpu_metrics = [
            m for m in tool_call.resource_metrics if m.type == ResourceType.CPU
        ]
        assert len(cpu_metrics) > 0
        assert cpu_metrics[0].value == 10.5

    def test_process_tool_call_metrics_with_error(self):
        """Test _process_tool_call_metrics with failed tool call."""
        response = MagicMock(spec=SandboxExecutionResponse)
        response.tool_calls = []

        call_data = {
            "tool_name": "failing_tool",
            "call_id": "call-789",
            "duration": 0.02,
            "success": False,
            "args": (),
            "kwargs": {"param": "value"},
            "error": "Tool execution failed",
            "start_time": "2026-01-01T12:00:00+00:00",
            "end_time": "2026-01-01T12:00:00.020+00:00",
        }

        metrics_per_tool_call = {"tool_calls": [call_data]}

        # Process metrics
        DockerSandbox._process_tool_call_metrics(response, metrics_per_tool_call)

        # Verify results
        assert len(response.tool_calls) == 1
        tool_call = response.tool_calls[0]

        assert tool_call.tool_name == "failing_tool"
        assert tool_call.success is False
        assert tool_call.error == "Tool execution failed"
        assert tool_call.input == {"param": "value"}
        assert tool_call.output is None

    def test_process_tool_call_metrics_empty(self):
        """Test _process_tool_call_metrics with empty metrics."""
        response = MagicMock(spec=SandboxExecutionResponse)
        response.tool_calls = []

        # Empty metrics should not cause issues
        DockerSandbox._process_tool_call_metrics(response, None)
        DockerSandbox._process_tool_call_metrics(response, {})
        DockerSandbox._process_tool_call_metrics(response, {"tool_calls": []})

        assert len(response.tool_calls) == 0

    def test_process_tool_call_metrics_with_invalid_data(self):
        """Test _process_tool_call_metrics handles invalid data gracefully."""
        response = MagicMock(spec=SandboxExecutionResponse)
        response.tool_calls = []

        # Invalid call data (missing required fields)
        invalid_call_data = {
            "tool_name": "invalid_tool"
            # Missing other required fields
        }

        metrics_per_tool_call = {"tool_calls": [invalid_call_data]}

        # Should handle gracefully without crashing
        DockerSandbox._process_tool_call_metrics(response, metrics_per_tool_call)

        # Should still create a tool call, even with incomplete data
        assert len(response.tool_calls) == 1
        tool_call = response.tool_calls[0]
        assert tool_call.tool_name == "invalid_tool"

    def test_build_tool_call_input_edge_cases(self):
        """Test _build_tool_call_input edge cases."""
        # Test with None values
        call_data = {"args": None, "kwargs": None}
        result = DockerSandbox._build_tool_call_input(call_data)
        assert result is None

        # Test with empty lists/dicts
        call_data = {"args": [], "kwargs": {}}
        result = DockerSandbox._build_tool_call_input(call_data)
        assert result is None

        # Test with mixed args and kwargs
        call_data = {
            "args": ("pos1", "pos2"),
            "kwargs": {"key1": "val1", "key2": "val2"},
        }
        result = DockerSandbox._build_tool_call_input(call_data)
        # Should prioritize kwargs over args
        assert result == {"key1": "val1", "key2": "val2"}

    def test_process_multiple_tool_calls(self):
        """Test processing multiple tool calls with parent-child relationships."""
        response = MagicMock(spec=SandboxExecutionResponse)
        response.tool_calls = []

        # Create parent and child calls
        parent_call = {
            "tool_name": "execute_python_code",
            "call_id": "parent-123",
            "duration": 0.1,
            "success": True,
            "args": (),
            "kwargs": {"code": "math.add(1, 2)"},
            "result": "Code executed",
            "start_time": "2026-01-01T12:00:00+00:00",
            "end_time": "2026-01-01T12:00:00.100+00:00",
        }

        child_call1 = {
            "tool_name": "math",
            "call_id": "child-456",
            "duration": 0.02,
            "success": True,
            "args": (),
            "kwargs": {"operation": "add", "numbers": [1, 2]},
            "result": 3,
            "start_time": "2026-01-01T12:00:00.010+00:00",
            "end_time": "2026-01-01T12:00:00.030+00:00",
        }

        child_call2 = {
            "tool_name": "final_answer",
            "call_id": "child-789",
            "duration": 0.01,
            "success": True,
            "args": (),
            "kwargs": {"answer": 3},
            "result": "The answer is 3",
            "start_time": "2026-01-01T12:00:00.040+00:00",
            "end_time": "2026-01-01T12:00:00.050+00:00",
        }

        metrics_per_tool_call = {"tool_calls": [parent_call, child_call1, child_call2]}

        # Process metrics
        DockerSandbox._process_tool_call_metrics(response, metrics_per_tool_call)

        # Verify all calls were processed
        assert len(response.tool_calls) == 3

        # Check parent call
        parent = next(c for c in response.tool_calls if c.call_id == "parent-123")
        assert parent.tool_name == "execute_python_code"
        assert parent.input == {"code": "math.add(1, 2)"}

        # Check child calls
        child1 = next(c for c in response.tool_calls if c.call_id == "child-456")
        assert child1.tool_name == "math"
        assert child1.input == {"operation": "add", "numbers": [1, 2]}
        assert child1.output == 3

        child2 = next(c for c in response.tool_calls if c.call_id == "child-789")
        assert child2.tool_name == "final_answer"
        assert child2.input == {"answer": 3}
        assert child2.output == "The answer is 3"
