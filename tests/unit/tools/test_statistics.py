"""
Tests for the statistics collection and tracking module.
"""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from local_coding_assistant.tools.statistics import (
    ResourceMetric,
    ResourceType,
    StatisticsManager,
    SystemStatistics,
    ToolCallMetrics,
    ToolStatistics,
)

# Test data
TEST_TOOL_NAME = "test_tool"
TEST_CALL_ID = "test_call_123"
TEST_START_TIME = datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC)
TEST_END_TIME = TEST_START_TIME + timedelta(seconds=1.5)
TEST_METRICS = [
    {"type": "cpu", "name": "cpu_usage", "value": 25.5, "unit": "percent"},
    {"type": "memory", "name": "memory_usage", "value": 1024, "unit": "MB"},
]


class TestResourceMetric:
    """Tests for the ResourceMetric class."""

    def test_creation(self):
        """Test creating a resource metric with all fields."""
        metric = ResourceMetric(
            type=ResourceType.CPU, name="test_metric", value=42.0, unit="test_units"
        )

        assert metric.type == ResourceType.CPU
        assert metric.name == "test_metric"
        assert metric.value == 42.0
        assert metric.unit == "test_units"
        assert isinstance(metric.timestamp, datetime)

    def test_creation_with_custom_timestamp(self):
        """Test creating a resource metric with a custom timestamp."""
        custom_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC)
        metric = ResourceMetric(
            type=ResourceType.MEMORY,
            name="test_metric",
            value=100,
            unit="MB",
            timestamp=custom_time,
        )

        assert metric.timestamp == custom_time


class TestToolCallMetrics:
    """Tests for the ToolCallMetrics class."""

    def test_creation(self):
        """Test creating a tool call metrics instance."""
        call_metrics = ToolCallMetrics(
            call_id=TEST_CALL_ID,
            tool_name=TEST_TOOL_NAME,
            start_time=TEST_START_TIME,
            end_time=TEST_END_TIME,
            duration=1.5,
            success=True,
        )

        assert call_metrics.call_id == TEST_CALL_ID
        assert call_metrics.tool_name == TEST_TOOL_NAME
        assert call_metrics.start_time == TEST_START_TIME
        assert call_metrics.end_time == TEST_END_TIME
        assert call_metrics.duration == 1.5
        assert call_metrics.success is True
        assert call_metrics.error is None
        assert call_metrics.metadata == {}

    def test_add_metric(self):
        """Test adding metrics to a tool call."""
        call_metrics = ToolCallMetrics(
            call_id=TEST_CALL_ID,
            tool_name=TEST_TOOL_NAME,
            start_time=TEST_START_TIME,
            end_time=TEST_END_TIME,
            duration=1.5,
            success=True,
        )

        metric = ResourceMetric(
            type=ResourceType.CPU, name="cpu_usage", value=42.0, unit="percent"
        )

        call_metrics.add_metric(metric)

        assert len(call_metrics.resource_metrics["cpu_usage"]) == 1
        assert call_metrics.resource_metrics["cpu_usage"][0] == metric

    def test_get_metric_stats(self):
        """Test getting statistics for a metric."""
        call_metrics = ToolCallMetrics(
            call_id=TEST_CALL_ID,
            tool_name=TEST_TOOL_NAME,
            start_time=TEST_START_TIME,
            end_time=TEST_END_TIME,
            duration=1.5,
            success=True,
        )

        # Add multiple metrics with the same name
        for i in range(1, 6):
            metric = ResourceMetric(
                type=ResourceType.CPU,
                name="cpu_usage",
                value=float(i * 10),
                unit="percent",
            )
            call_metrics.add_metric(metric)

        stats = call_metrics.get_metric_stats("cpu_usage")

        assert stats is not None
        assert stats["min"] == 10.0
        assert stats["max"] == 50.0
        assert stats["avg"] == 30.0
        assert stats["count"] == 5
        assert stats["total"] == 150.0
        assert len(stats["values"]) == 5
        assert len(stats["timestamps"]) == 5


class TestToolStatistics:
    """Tests for the ToolStatistics class."""

    def test_initial_state(self):
        """Test the initial state of ToolStatistics."""
        stats = ToolStatistics(TEST_TOOL_NAME)

        assert stats.tool_name == TEST_TOOL_NAME
        assert stats.total_executions == 0
        assert stats.success_count == 0
        assert stats.error_count == 0
        assert stats.total_duration == 0.0
        assert stats.avg_duration == 0.0
        assert stats.last_execution is None
        assert stats.first_execution is None
        assert stats.last_error is None
        assert stats.recent_calls == []

    def test_record_successful_call(self):
        """Test recording a successful tool call."""
        stats = ToolStatistics(TEST_TOOL_NAME)
        call = ToolCallMetrics(
            call_id=TEST_CALL_ID,
            tool_name=TEST_TOOL_NAME,
            start_time=TEST_START_TIME,
            end_time=TEST_END_TIME,
            duration=1.5,
            success=True,
        )

        stats.record_call(call)

        assert stats.total_executions == 1
        assert stats.success_count == 1
        assert stats.error_count == 0
        assert stats.total_duration == 1.5
        assert stats.avg_duration == 1.5
        assert stats.last_execution == TEST_END_TIME
        assert stats.first_execution == TEST_START_TIME
        assert stats.last_error is None
        assert len(stats.recent_calls) == 1
        assert stats.recent_calls[0] == call

    def test_record_failed_call(self):
        """Test recording a failed tool call."""
        stats = ToolStatistics(TEST_TOOL_NAME)
        error_msg = "Test error"
        call = ToolCallMetrics(
            call_id=TEST_CALL_ID,
            tool_name=TEST_TOOL_NAME,
            start_time=TEST_START_TIME,
            end_time=TEST_END_TIME,
            duration=0.5,
            success=False,
            error=error_msg,
        )

        stats.record_call(call)

        assert stats.total_executions == 1
        assert stats.success_count == 0
        assert stats.error_count == 1
        assert stats.last_error == error_msg

    def test_metric_statistics(self):
        """Test recording and retrieving metric statistics."""
        stats = ToolStatistics(TEST_TOOL_NAME)

        # First call with metrics
        call1 = ToolCallMetrics(
            call_id=TEST_CALL_ID + "_1",
            tool_name=TEST_TOOL_NAME,
            start_time=TEST_START_TIME,
            end_time=TEST_START_TIME + timedelta(seconds=1),
            duration=1.0,
            success=True,
        )

        metric1 = ResourceMetric(
            type=ResourceType.CPU, name="cpu_usage", value=20.0, unit="percent"
        )
        call1.add_metric(metric1)

        # Second call with metrics
        call2 = ToolCallMetrics(
            call_id=TEST_CALL_ID + "_2",
            tool_name=TEST_TOOL_NAME,
            start_time=TEST_START_TIME + timedelta(seconds=1),
            end_time=TEST_START_TIME + timedelta(seconds=3),
            duration=2.0,
            success=True,
        )

        metric2 = ResourceMetric(
            type=ResourceType.CPU, name="cpu_usage", value=40.0, unit="percent"
        )
        call2.add_metric(metric2)

        # Record both calls
        stats.record_call(call1)
        stats.record_call(call2)

        # Check metric statistics
        metric_stats = stats.get_metric_stats("cpu_usage")

        assert metric_stats is not None
        assert metric_stats["min"] == 20.0
        assert metric_stats["max"] == 40.0
        assert metric_stats["avg"] == 30.0
        assert metric_stats["count"] == 2
        assert metric_stats["total"] == 60.0
        assert len(metric_stats["values"]) == 2
        assert len(metric_stats["timestamps"]) == 2

        # Test metrics summary
        summary = stats.get_metrics_summary()
        assert "cpu_usage" in summary
        assert summary["cpu_usage"]["min"] == 20.0
        assert summary["cpu_usage"]["max"] == 40.0
        assert summary["cpu_usage"]["avg"] == 30.0

    def test_recent_calls_limit(self):
        """Test that recent calls are limited to max_recent_calls."""
        max_calls = 5
        stats = ToolStatistics(TEST_TOOL_NAME, max_recent_calls=max_calls)

        # Add more calls than the limit
        for i in range(max_calls + 2):
            call = ToolCallMetrics(
                call_id=f"{TEST_CALL_ID}_{i}",
                tool_name=TEST_TOOL_NAME,
                start_time=TEST_START_TIME + timedelta(seconds=i),
                end_time=TEST_START_TIME + timedelta(seconds=i + 1),
                duration=1.0,
                success=True,
            )
            stats.record_call(call)

        # Should only keep the most recent calls up to max_recent_calls
        assert len(stats.recent_calls) == max_calls
        # The most recent call should be the last one added (i = max_calls + 1)
        assert stats.recent_calls[-1].call_id == f"{TEST_CALL_ID}_{max_calls + 1}"
        # The oldest call in the recent calls should be the first one that wasn't discarded
        assert (
            stats.recent_calls[0].call_id == f"{TEST_CALL_ID}_2"
        )  # 2 = (max_calls + 2) - max_calls


class TestSystemStatistics:
    """Tests for the SystemStatistics class."""

    def test_initial_state(self):
        """Test the initial state of SystemStatistics."""
        stats = SystemStatistics()

        assert stats.total_executions == 0
        assert stats.total_duration == 0.0
        assert stats.avg_duration == 0.0
        assert stats.last_execution is None
        assert stats.first_execution is None
        assert stats.system_metrics == {}

    def test_record_execution(self):
        """Test recording a system execution with metrics."""
        stats = SystemStatistics()
        start_time = datetime.now(UTC)

        metrics = [
            ResourceMetric(
                type=ResourceType.CPU, name="cpu_usage", value=25.5, unit="percent"
            ),
            ResourceMetric(
                type=ResourceType.MEMORY, name="memory_usage", value=1024, unit="MB"
            ),
        ]

        stats.record_execution(1.5, metrics)

        assert stats.total_executions == 1
        assert stats.total_duration == 1.5
        assert stats.avg_duration == 1.5
        assert stats.last_execution is not None
        assert stats.first_execution == stats.last_execution

        # Check that metrics were recorded
        assert "cpu_usage" in stats.system_metrics
        assert "memory_usage" in stats.system_metrics
        assert len(stats.system_metrics["cpu_usage"]) == 1
        assert len(stats.system_metrics["memory_usage"]) == 1

        # Check metric statistics
        cpu_stats = stats.get_metric_stats("cpu_usage")
        assert cpu_stats is not None
        assert cpu_stats["min"] == 25.5
        assert cpu_stats["max"] == 25.5
        assert cpu_stats["avg"] == 25.5
        assert cpu_stats["count"] == 1

        # Record another execution with updated metrics
        metrics = [
            ResourceMetric(
                type=ResourceType.CPU, name="cpu_usage", value=30.0, unit="percent"
            ),
            ResourceMetric(
                type=ResourceType.MEMORY, name="memory_usage", value=1500, unit="MB"
            ),
        ]

        stats.record_execution(2.0, metrics)

        # Check updated statistics
        assert stats.total_executions == 2
        assert stats.total_duration == 3.5
        assert stats.avg_duration == 1.75

        # Check updated metric statistics
        cpu_stats = stats.get_metric_stats("cpu_usage")
        assert cpu_stats is not None
        assert cpu_stats["min"] == 25.5
        assert cpu_stats["max"] == 30.0
        assert cpu_stats["avg"] == 27.75
        assert cpu_stats["count"] == 2


class TestStatisticsManager:
    """Tests for the StatisticsManager class."""

    @pytest.fixture
    def manager(self):
        """Create a new StatisticsManager instance for testing."""
        return StatisticsManager()

    @pytest.mark.asyncio
    async def test_record_tool_call(self, manager):
        """Test recording a tool call with metrics."""
        await manager.record_tool_call(
            tool_name=TEST_TOOL_NAME,
            call_id=TEST_CALL_ID,
            start_time=TEST_START_TIME,
            end_time=TEST_END_TIME,
            success=True,
            resource_metrics=TEST_METRICS,
            metadata={"test": "value"},
        )

        # Check that the tool statistics were updated
        tool_stats = manager.get_tool_stats(TEST_TOOL_NAME)
        assert tool_stats is not None
        assert tool_stats.tool_name == TEST_TOOL_NAME
        assert tool_stats.total_executions == 1
        assert tool_stats.success_count == 1
        assert tool_stats.error_count == 0
        assert tool_stats.total_duration == 1.5
        assert tool_stats.avg_duration == 1.5
        assert tool_stats.last_execution == TEST_END_TIME
        assert tool_stats.first_execution == TEST_START_TIME

        # Check that the call was recorded in recent calls
        assert len(tool_stats.recent_calls) == 1
        call_metrics = tool_stats.recent_calls[0]
        assert call_metrics.call_id == TEST_CALL_ID
        assert call_metrics.tool_name == TEST_TOOL_NAME
        assert call_metrics.metadata == {"test": "value"}

        # Check that metrics were recorded
        cpu_stats = tool_stats.get_metric_stats("cpu_usage")
        assert cpu_stats is not None
        assert cpu_stats["min"] == 25.5

        memory_stats = tool_stats.get_metric_stats("memory_usage")
        assert memory_stats is not None
        assert memory_stats["min"] == 1024

    @pytest.mark.asyncio
    async def test_record_system_metrics(self, manager):
        """Test recording system metrics."""
        system_metrics = [
            {"type": "cpu", "name": "system_cpu", "value": 50.0, "unit": "percent"},
            {"type": "memory", "name": "system_memory", "value": 2048, "unit": "MB"},
        ]

        await manager.record_system_metrics(
            metrics=system_metrics, execution_duration=1.0
        )

        # Check that system statistics were updated
        system_stats = manager.get_system_stats()
        assert system_stats.total_executions == 1
        assert system_stats.total_duration == 1.0
        assert system_stats.avg_duration == 1.0

        # Check that metrics were recorded
        cpu_stats = system_stats.get_metric_stats("system_cpu")
        assert cpu_stats is not None
        assert cpu_stats["min"] == 50.0

        memory_stats = system_stats.get_metric_stats("system_memory")
        assert memory_stats is not None
        assert memory_stats["min"] == 2048

    @pytest.mark.asyncio
    async def test_reset_tool_stats(self, manager):
        """Test resetting tool statistics."""
        # Record a tool call
        await manager.record_tool_call(
            tool_name=TEST_TOOL_NAME,
            call_id=TEST_CALL_ID,
            start_time=TEST_START_TIME,
            end_time=TEST_END_TIME,
            success=True,
            resource_metrics=TEST_METRICS,
        )

        # Reset the tool stats
        manager.reset_tool_stats(TEST_TOOL_NAME)

        # Check that the tool stats were reset
        tool_stats = manager.get_tool_stats(TEST_TOOL_NAME)
        assert tool_stats.total_executions == 0
        assert len(tool_stats.recent_calls) == 0

    @pytest.mark.asyncio
    async def test_reset_system_stats(self, manager):
        """Test resetting system statistics."""
        # Record system metrics
        await manager.record_system_metrics(
            metrics=[
                {"type": "cpu", "name": "system_cpu", "value": 50.0, "unit": "percent"}
            ],
            execution_duration=1.0,
        )

        # Reset system stats
        manager.reset_system_stats()

        # Check that system stats were reset
        system_stats = manager.get_system_stats()
        assert system_stats.total_executions == 0
        assert system_stats.total_duration == 0.0
        assert system_stats.avg_duration == 0.0

    @pytest.mark.asyncio
    async def test_concurrent_access(self, manager):
        """Test that the StatisticsManager is thread-safe."""
        num_calls = 10

        async def record_tool_call(i):
            await manager.record_tool_call(
                tool_name=f"tool_{i % 2}",
                call_id=f"call_{i}",
                start_time=TEST_START_TIME + timedelta(seconds=i),
                end_time=TEST_START_TIME + timedelta(seconds=i + 1),
                success=True,
            )

        # Record multiple tool calls concurrently
        tasks = [record_tool_call(i) for i in range(num_calls)]
        await asyncio.gather(*tasks)

        # Check that all calls were recorded
        tool0_stats = manager.get_tool_stats("tool_0")
        tool1_stats = manager.get_tool_stats("tool_1")

        assert tool0_stats.total_executions + tool1_stats.total_executions == num_calls
        assert tool0_stats.total_executions in [num_calls // 2, (num_calls + 1) // 2]
        assert tool1_stats.total_executions in [num_calls // 2, (num_calls + 1) // 2]

        # Check that the lock was released properly
        assert manager._lock.locked() is False
