"""
Statistics collection and tracking for tool execution with detailed resource metrics.
"""

from __future__ import annotations

import asyncio
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, TypedDict

from pydantic import BaseModel, Field


class ResourceType(str, Enum):
    """Type of resource being measured."""

    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    DISK = "disk"
    CUSTOM = "custom"


class MetricStats(TypedDict):
    """Statistics for a specific metric type."""

    min: float
    max: float
    avg: float
    count: int
    total: float
    values: list[float]
    timestamps: list[datetime]


class ResourceMetric(BaseModel):
    """Base class for resource metrics."""

    type: ResourceType
    name: str
    value: float | int | dict[str, Any]
    unit: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


@dataclass
class ToolCallMetrics:
    """Metrics for a single tool call execution."""

    call_id: str
    tool_name: str
    start_time: datetime
    end_time: datetime
    duration: float  # in seconds
    success: bool
    error: str | None = None
    resource_metrics: dict[str, list[ResourceMetric]] = field(
        default_factory=lambda: defaultdict(list)
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_metric(self, metric: ResourceMetric) -> None:
        """Add a resource metric to this tool call."""
        self.resource_metrics[metric.name].append(metric)

    def get_metric_stats(self, metric_name: str) -> MetricStats | None:
        """Get statistics for a specific metric."""
        metrics = self.resource_metrics.get(metric_name, [])
        if not metrics:
            return None

        values = [
            float(m.value) if isinstance(m.value, (int, float)) else 0 for m in metrics
        ]
        return MetricStats(
            min=min(values) if values else 0,
            max=max(values) if values else 0,
            avg=statistics.mean(values) if values else 0,
            count=len(values),
            total=sum(values),
            values=values,
            timestamps=[m.timestamp for m in metrics],
        )


@dataclass
class ToolStatistics:
    """Statistics for a specific tool's executions."""

    tool_name: str
    total_executions: int = 0
    success_count: int = 0
    error_count: int = 0
    total_duration: float = 0.0
    avg_duration: float = 0.0
    last_execution: datetime | None = None
    first_execution: datetime | None = None
    last_error: str | None = None
    recent_calls: list[ToolCallMetrics] = field(default_factory=list)
    max_recent_calls: int = 100
    _metric_stats: dict[str, MetricStats] = field(default_factory=dict)

    def record_call(self, call: ToolCallMetrics) -> None:
        """Record a new tool call."""
        self.total_executions += 1
        self.total_duration += call.duration
        self.avg_duration = self.total_duration / self.total_executions
        self.last_execution = call.end_time or datetime.now(UTC)

        if not self.first_execution:
            self.first_execution = call.start_time

        if call.success:
            self.success_count += 1
        else:
            self.error_count += 1
            self.last_error = call.error

        # Add to recent calls
        self.recent_calls.append(call)
        if len(self.recent_calls) > self.max_recent_calls:
            self.recent_calls = self.recent_calls[-self.max_recent_calls :]

        # Update metric statistics
        self._update_metric_stats(call)

    def _update_metric_stats(self, call: ToolCallMetrics) -> None:
        """Update statistics for all metrics in the call."""
        for metric_name, metrics in call.resource_metrics.items():
            values = [
                float(m.value) if isinstance(m.value, (int, float)) else 0
                for m in metrics
            ]
            if not values:
                continue

            if metric_name not in self._metric_stats:
                self._metric_stats[metric_name] = MetricStats(
                    min=values[0],
                    max=values[0],
                    avg=values[0],
                    count=1,
                    total=values[0],
                    values=values[:],
                    timestamps=[m.timestamp for m in metrics],
                )
            else:
                stats = self._metric_stats[metric_name]
                stats["min"] = min(stats["min"], min(values))
                stats["max"] = max(stats["max"], max(values))
                stats["count"] += len(values)
                stats["total"] += sum(values)
                stats["avg"] = stats["total"] / stats["count"]
                stats["values"].extend(values)
                stats["timestamps"].extend(m.timestamp for m in metrics)

    def get_metric_stats(self, metric_name: str) -> MetricStats | None:
        """Get statistics for a specific metric across all calls."""
        return self._metric_stats.get(metric_name)

    @property
    def success_rate(self) -> float:
        """Calculate the success rate as a percentage."""
        if self.total_executions == 0:
            return 0.0
        return (self.success_count / self.total_executions) * 100

    def get_metrics_summary(self) -> dict[str, dict[str, float]]:
        """Get a summary of all metrics with their statistics."""
        return {
            name: {
                "min": stats["min"],
                "max": stats["max"],
                "avg": stats["avg"],
                "count": stats["count"],
            }
            for name, stats in self._metric_stats.items()
        }


@dataclass
class SystemStatistics:
    """System-wide execution statistics."""

    total_executions: int = 0
    total_duration: float = 0.0
    avg_duration: float = 0.0
    last_execution: datetime | None = None
    first_execution: datetime | None = None
    system_metrics: dict[str, list[ResourceMetric]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _metric_stats: dict[str, MetricStats] = field(default_factory=dict)

    def record_execution(self, duration: float, metrics: list[ResourceMetric]) -> None:
        """Record a system execution with its metrics."""
        now = datetime.now(UTC)
        self.total_executions += 1
        self.total_duration += duration
        self.avg_duration = self.total_duration / self.total_executions

        if not self.first_execution:
            self.first_execution = now
        self.last_execution = now

        # Record all metrics
        for metric in metrics:
            self.system_metrics[metric.name].append(metric)

            # Update metric statistics
            value = float(metric.value) if isinstance(metric.value, (int, float)) else 0
            if metric.name not in self._metric_stats:
                self._metric_stats[metric.name] = MetricStats(
                    min=value,
                    max=value,
                    avg=value,
                    count=1,
                    total=value,
                    values=[value],
                    timestamps=[metric.timestamp],
                )
            else:
                stats = self._metric_stats[metric.name]
                stats["min"] = min(stats["min"], value)
                stats["max"] = max(stats["max"], value)
                stats["count"] += 1
                stats["total"] += value
                stats["avg"] = stats["total"] / stats["count"]
                stats["values"].append(value)
                stats["timestamps"].append(metric.timestamp)

    def get_metric_stats(self, metric_name: str) -> MetricStats | None:
        """Get statistics for a specific system metric."""
        return self._metric_stats.get(metric_name)

    def get_metrics_summary(self) -> dict[str, dict[str, float]]:
        """Get a summary of all system metrics with their statistics."""
        return {
            name: {
                "min": stats["min"],
                "max": stats["max"],
                "avg": stats["avg"],
                "count": stats["count"],
            }
            for name, stats in self._metric_stats.items()
        }


class StatisticsManager:
    """Centralized manager for all execution statistics."""

    def __init__(self):
        self._tool_stats: dict[str, ToolStatistics] = {}
        self._system_stats = SystemStatistics()
        self._lock = asyncio.Lock()

    async def record_tool_call(
        self,
        tool_name: str,
        call_id: str,
        start_time: datetime,
        end_time: datetime,
        success: bool,
        error: str | None = None,
        resource_metrics: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a single tool call with its metrics.

        Args:
            tool_name: Name of the tool being called
            call_id: Unique identifier for this call
            start_time: When the call started
            end_time: When the call completed
            success: Whether the call was successful
            error: Error message if the call failed
            resource_metrics: List of resource metrics for this call
            metadata: Additional metadata about the call
        """
        async with self._lock:
            if tool_name not in self._tool_stats:
                self._tool_stats[tool_name] = ToolStatistics(tool_name)

            duration = (end_time - start_time).total_seconds()
            call_metrics = ToolCallMetrics(
                call_id=call_id,
                tool_name=tool_name,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                success=success,
                error=error,
                metadata=metadata or {},
            )

            # Add all resource metrics to the call
            for metric_data in resource_metrics or []:
                try:
                    # Handle timestamp - it might already be a datetime object or a string
                    timestamp = metric_data.get("timestamp")
                    if timestamp is None:
                        timestamp = datetime.now(UTC)
                    elif isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp)

                    metric = ResourceMetric(
                        type=ResourceType(metric_data.get("type", "custom")),
                        name=metric_data["name"],
                        value=metric_data["value"],
                        unit=metric_data.get("unit", "count"),
                        timestamp=timestamp,
                    )
                    call_metrics.add_metric(metric)
                except (KeyError, ValueError):
                    continue  # Skip invalid metrics

            # Record the call in tool statistics
            self._tool_stats[tool_name].record_call(call_metrics)

    async def record_system_metrics(
        self, metrics: list[dict[str, Any]], execution_duration: float
    ) -> None:
        """Record system-level metrics for an execution.

        Args:
            metrics: List of system metrics
            execution_duration: Total duration of the execution in seconds
        """
        resource_metrics = []

        for metric_data in metrics:
            try:
                # Handle timestamp - it might already be a datetime object or a string
                timestamp = metric_data.get("timestamp")
                if timestamp is None:
                    timestamp = datetime.now(UTC)
                elif isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)

                metric = ResourceMetric(
                    type=ResourceType(metric_data.get("type", "custom")),
                    name=metric_data["name"],
                    value=metric_data["value"],
                    unit=metric_data.get("unit", "count"),
                    timestamp=timestamp,
                )
                resource_metrics.append(metric)
            except (KeyError, ValueError):
                continue  # Skip invalid metrics

        async with self._lock:
            self._system_stats.record_execution(execution_duration, resource_metrics)

    def get_tool_stats(
        self, tool_name: str | None = None
    ) -> ToolStatistics | dict[str, ToolStatistics]:
        """Get statistics for a specific tool or all tools.

        Args:
            tool_name: Optional name of the tool to get stats for

        Returns:
            Either a ToolStatistics object or a dictionary of all tools' statistics
        """
        if tool_name:
            return self._tool_stats.get(tool_name, ToolStatistics(tool_name))
        return dict(self._tool_stats)

    def get_system_stats(self) -> SystemStatistics:
        """Get system-wide statistics."""
        return self._system_stats

    def reset_tool_stats(self, tool_name: str | None = None) -> None:
        """Reset statistics for a specific tool or all tools."""
        if tool_name:
            if tool_name in self._tool_stats:
                self._tool_stats[tool_name] = ToolStatistics(tool_name)
        else:
            self._tool_stats.clear()

    def reset_system_stats(self) -> None:
        """Reset system statistics."""
        self._system_stats = SystemStatistics()

    def reset_all(self) -> None:
        """Reset all statistics."""
        self.reset_tool_stats()
        self.reset_system_stats()

    def get_metric_history(
        self,
        metric_name: str,
        tool_name: str | None = None,
        time_range: tuple[datetime, datetime] | None = None,
    ) -> list[dict[str, Any]]:
        """Get historical data for a specific metric.

        Args:
            metric_name: Name of the metric to retrieve
            tool_name: Optional tool name to filter by
            time_range: Optional (start, end) datetime range to filter by

        Returns:
            List of metric values with timestamps
        """
        results = []

        if tool_name:
            tools = (
                [self._tool_stats["tool_name"]] if tool_name in self._tool_stats else []
            )
        else:
            tools = list(self._tool_stats.values())

        for tool in tools:
            for call in tool.recent_calls:
                for metric_list in call.resource_metrics.values():
                    for metric in metric_list:
                        if metric.name == metric_name:
                            if time_range:
                                start, end = time_range
                                if not (start <= metric.timestamp <= end):
                                    continue
                            results.append(
                                {
                                    "tool": tool.tool_name,
                                    "call_id": call.call_id,
                                    "timestamp": metric.timestamp,
                                    "value": metric.value,
                                    "unit": metric.unit,
                                }
                            )

        # Sort by timestamp
        return sorted(results, key=lambda x: x["timestamp"])
