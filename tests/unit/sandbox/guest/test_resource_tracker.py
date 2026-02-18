"""Tests for the updated resource tracker with proper argument capture."""

import pytest
import asyncio
import time
from unittest.mock import MagicMock, patch

from local_coding_assistant.sandbox.guest.resource_tracker import ResourceTracker


class TestResourceTracker:
    """Test ResourceTracker with enhanced argument capture."""

    def test_tracker_with_explicit_args_kwargs(self):
        """Test tracker decorator with explicit args and kwargs."""
        tracker = ResourceTracker()

        # Test data
        test_args = ("arg1", "arg2")
        test_kwargs = {"param1": "value1", "param2": "value2"}

        @tracker.track(tool_name="test_tool", args=test_args, kwargs=test_kwargs)
        async def test_async_func():
            return "test_result"

        @tracker.track(tool_name="test_tool_sync", args=test_args, kwargs=test_kwargs)
        def test_sync_func():
            return "test_result"

        # Test async function
        result = asyncio.run(test_async_func())
        assert result == "test_result"

        # Check metrics
        metrics = tracker.get_metrics()
        assert metrics is not None
        assert len(metrics["tool_calls"]) == 1

        call_data = metrics["tool_calls"][0]
        assert call_data["tool_name"] == "test_tool"
        assert call_data["args"] == test_args
        assert call_data["kwargs"] == test_kwargs
        assert call_data["success"] is True
        assert call_data["result"] == "test_result"

        # Reset for sync test
        tracker.reset()

        # Test sync function
        result = test_sync_func()
        assert result == "test_result"

        # Check metrics
        metrics = tracker.get_metrics()
        assert metrics is not None
        assert len(metrics["tool_calls"]) == 1

        call_data = metrics["tool_calls"][0]
        assert call_data["tool_name"] == "test_tool_sync"
        assert call_data["args"] == test_args
        assert call_data["kwargs"] == test_kwargs
        assert call_data["success"] is True
        assert call_data["result"] == "test_result"

    def test_tracker_with_default_args_kwargs(self):
        """Test tracker decorator with default empty args/kwargs."""
        tracker = ResourceTracker()

        @tracker.track(tool_name="test_tool_default")
        async def test_async_func():
            return "test_result"

        # Execute
        result = asyncio.run(test_async_func())
        assert result == "test_result"

        # Check metrics - should use provided args/kwargs (empty by default)
        metrics = tracker.get_metrics()
        assert metrics is not None
        assert len(metrics["tool_calls"]) == 1

        call_data = metrics["tool_calls"][0]
        assert call_data["tool_name"] == "test_tool_default"
        assert call_data["args"] == ()
        assert call_data["kwargs"] == {}
        assert call_data["success"] is True

    def test_tracker_with_function_args_kwargs(self):
        """Test that function args/kwargs are ignored when explicit ones provided."""
        tracker = ResourceTracker()

        explicit_args = ("explicit_arg",)
        explicit_kwargs = {"explicit_param": "explicit_value"}

        @tracker.track(
            tool_name="test_tool", args=explicit_args, kwargs=explicit_kwargs
        )
        async def test_async_func(func_arg1, func_arg2, func_kwarg1=None):
            return f"{func_arg1}_{func_arg2}_{func_kwarg1}"

        # Execute with different function arguments
        result = asyncio.run(
            test_async_func("func_val1", "func_val2", func_kwarg1="func_val3")
        )
        assert result == "func_val1_func_val2_func_val3"

        # Check metrics - should use explicit args/kwargs, not function args/kwargs
        metrics = tracker.get_metrics()
        assert metrics is not None
        assert len(metrics["tool_calls"]) == 1

        call_data = metrics["tool_calls"][0]
        assert call_data["tool_name"] == "test_tool"
        assert call_data["args"] == explicit_args
        assert call_data["kwargs"] == explicit_kwargs
        # Should NOT be the function arguments
        assert call_data["args"] != ("func_val1", "func_val2")
        assert call_data["kwargs"] != {"func_kwarg1": "func_val3"}

    def test_tracker_error_handling(self):
        """Test tracker error handling with explicit args/kwargs."""
        tracker = ResourceTracker()

        test_args = ("arg1",)
        test_kwargs = {"param1": "value1"}

        @tracker.track(tool_name="test_tool_error", args=test_args, kwargs=test_kwargs)
        async def failing_func():
            raise ValueError("Test error")

        # Execute and expect error
        with pytest.raises(ValueError, match="Test error"):
            asyncio.run(failing_func())

        # Check metrics
        metrics = tracker.get_metrics()
        assert metrics is not None
        assert len(metrics["tool_calls"]) == 1

        call_data = metrics["tool_calls"][0]
        assert call_data["tool_name"] == "test_tool_error"
        assert call_data["args"] == test_args
        assert call_data["kwargs"] == test_kwargs
        assert call_data["success"] is False
        assert "Test error" in call_data["error"]

    @patch("local_coding_assistant.sandbox.guest.resource_tracker.psutil.Process")
    def test_tracker_resource_metrics(self, mock_process):
        """Test that resource metrics are still captured properly."""
        # Mock psutil Process
        mock_proc = MagicMock()
        mock_proc.memory_info.return_value = MagicMock(rss=1024 * 1024 * 100)  # 100MB
        mock_proc.cpu_percent.return_value = 25.0
        mock_proc.io_counters.return_value = MagicMock(read_bytes=1000, write_bytes=500)
        mock_process.return_value = mock_proc

        tracker = ResourceTracker()

        test_args = ("arg1",)
        test_kwargs = {"param1": "value1"}

        @tracker.track(
            tool_name="test_tool_metrics", args=test_args, kwargs=test_kwargs
        )
        async def test_func():
            time.sleep(0.01)  # Small delay to ensure duration > 0
            return "result"

        # Execute
        result = asyncio.run(test_func())
        assert result == "result"

        # Check metrics
        metrics = tracker.get_metrics()
        assert metrics is not None
        assert len(metrics["tool_calls"]) == 1

        call_data = metrics["tool_calls"][0]
        assert call_data["tool_name"] == "test_tool_metrics"
        assert call_data["args"] == test_args
        assert call_data["kwargs"] == test_kwargs
        assert call_data["success"] is True

        # Check resource metrics
        assert "start_stats" in call_data
        assert "end_stats" in call_data
        assert "delta_stats" in call_data
        assert call_data["duration"] > 0

    def test_tracker_multiple_calls(self):
        """Test tracker with multiple calls using different args/kwargs."""
        tracker = ResourceTracker()

        # Reset and test multiple calls with same decorator
        tracker.reset()

        @tracker.track(
            tool_name="multi_tool", args=("call1",), kwargs={"param": "value1"}
        )
        async def test_func1():
            return "result1"

        @tracker.track(
            tool_name="multi_tool", args=("call2",), kwargs={"param": "value2"}
        )
        async def test_func2():
            return "result2"

        result1 = asyncio.run(test_func1())
        result2 = asyncio.run(test_func2())

        assert result1 == "result1"
        assert result2 == "result2"

        # Check metrics - should have both calls
        metrics = tracker.get_metrics()
        assert metrics is not None
        assert len(metrics["tool_calls"]) == 2

        # First call
        call1 = next(
            c for c in metrics["tool_calls"] if c["kwargs"]["param"] == "value1"
        )
        assert call1["args"] == ("call1",)

        # Second call
        call2 = next(
            c for c in metrics["tool_calls"] if c["kwargs"]["param"] == "value2"
        )
        assert call2["args"] == ("call2",)
