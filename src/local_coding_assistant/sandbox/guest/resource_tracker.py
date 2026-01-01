"""Resource tracking for sandbox tool executions."""

import asyncio
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast

import psutil

T = TypeVar("T", bound=Callable[..., Any])


class ResourceTracker:
    """Tracks resource usage and results for tool executions."""

    def __init__(self):
        self.metrics: dict[str, Any] = {
            "tool_calls": [],
            "total_duration": 0.0,
            "success_count": 0,
            "error_count": 0,
            "results": {},  # Store results by call ID
            "tool_results": {},  # Store last result by tool name
        }
        self._process = psutil.Process()
        self._prev_io = None
        self._prev_time = None
        self._lock = asyncio.Lock()  # For thread-safe operations

    def _get_container_stats(self) -> dict[str, float]:
        """Get current container resource usage stats.

        Returns:
            Dict containing the most relevant resource metrics:
            - memory_rss_mb: Resident Set Size in MB (shows actual RAM usage)
            - cpu_percent: CPU usage as a percentage
            - read_bytes_per_sec: Disk read speed in bytes/second
            - write_bytes_per_sec: Disk write speed in bytes/second
        """
        # Get memory usage
        mem = self._process.memory_info()

        # Get CPU usage (using interval=None for non-blocking call)
        cpu_percent = self._process.cpu_percent(interval=None)

        # Get I/O stats
        io = (
            self._process.io_counters()
            if hasattr(self._process, "io_counters") and self._process
            else None
        )
        current_time = time.time()

        io_stats = {}
        if self._prev_io is not None and self._prev_time is not None and io is not None:
            time_delta = max(
                0.001, current_time - self._prev_time
            )  # Prevent division by zero
            io_stats = {
                "read_bytes_per_sec": (io.read_bytes - self._prev_io.read_bytes)
                / time_delta,
                "write_bytes_per_sec": (io.write_bytes - self._prev_io.write_bytes)
                / time_delta,
            }

        # Update previous values for next call
        self._prev_io = io
        self._prev_time = current_time

        return {
            "memory_rss_mb": mem.rss / (1024 * 1024),  # Convert to MB
            "cpu_percent": cpu_percent,
            **io_stats,
        }

    def track(self, tool_name: str | None = None) -> Callable[[T], T]:
        """Decorator to track resource usage of a tool function."""

        def decorator(func: T) -> T:
            nonlocal tool_name
            name = tool_name or getattr(func, "__name__", "unknown")

            if asyncio.iscoroutinefunction(func):

                @wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    start_time = time.time()
                    start_stats = self._get_container_stats()

                    try:
                        result = await func(*args, **kwargs)
                        end_stats = self._get_container_stats()
                        self.record_metrics(
                            tool_name=name,
                            duration=time.time() - start_time,
                            success=True,
                            args=args,
                            kwargs=kwargs,
                            start_stats=start_stats,
                            end_stats=end_stats,
                            result=result,
                        )
                        return result
                    except Exception as e:
                        end_stats = self._get_container_stats()
                        self.record_metrics(
                            tool_name=name,
                            duration=time.time() - start_time,
                            success=False,
                            args=args,
                            kwargs=kwargs,
                            start_stats=start_stats,
                            end_stats=end_stats,
                            error=e,
                        )
                        raise

                # This cast is safe because we're preserving the function signature
                return cast(T, async_wrapper)
            else:

                @wraps(func)
                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    start_time = time.time()
                    start_stats = self._get_container_stats()

                    try:
                        result = func(*args, **kwargs)
                        end_stats = self._get_container_stats()
                        self.record_metrics(
                            tool_name=name,
                            duration=time.time() - start_time,
                            success=True,
                            args=args,
                            kwargs=kwargs,
                            start_stats=start_stats,
                            end_stats=end_stats,
                            result=result,
                        )
                        return result
                    except Exception as e:
                        end_stats = self._get_container_stats()
                        self.record_metrics(
                            tool_name=name,
                            duration=time.time() - start_time,
                            success=False,
                            args=args,
                            kwargs=kwargs,
                            start_stats=start_stats,
                            end_stats=end_stats,
                            error=e,
                        )
                        raise

                # This cast is safe because we're preserving the function signature
                return cast(T, sync_wrapper)

        return decorator

    def record_metrics(
        self,
        tool_name: str,
        duration: float,
        success: bool,
        args: tuple,
        kwargs: dict,
        start_stats: dict[str, float],
        end_stats: dict[str, float],
        result: Any = None,
        error: Exception | None = None,
    ) -> str:
        """Record execution metrics with container stats and result.

        Args:
            tool_name: Name of the tool being executed
            duration: Execution duration in seconds
            success: Whether the execution was successful
            args: Positional arguments passed to the tool
            kwargs: Keyword arguments passed to the tool
            start_stats: Resource stats at start of execution
            end_stats: Resource stats at end of execution
            result: Result of the tool execution (if successful)
            error: Exception if execution failed

        Returns:
            str: Unique ID for this tool call
        """
        call_id = str(time.time_ns())  # Simple unique ID
        metrics = {
            "id": call_id,
            "tool_name": tool_name,
            "duration": duration,
            "success": success,
            "timestamp": time.time(),
            "args": args,
            "kwargs": kwargs,
            "start_stats": start_stats,
            "end_stats": end_stats,
            "delta_stats": {
                "cpu_delta": end_stats.get("cpu_percent", 0)
                - start_stats.get("cpu_percent", 0),
                "memory_delta_mb": end_stats.get("memory_rss_mb", 0)
                - start_stats.get("memory_rss_mb", 0),
                **{
                    k: v
                    for k, v in end_stats.items()
                    if k.startswith("read_") or k.startswith("write_")
                },
            },
            "result": result if success else None,
            "error": str(error) if error else None,
        }

        # Update metrics synchronously
        # Use a regular lock instead of asyncio.Lock for synchronous context
        if not hasattr(self, "_sync_lock"):
            import threading

            self._sync_lock = threading.Lock()

        with self._sync_lock:
            self.metrics["tool_calls"].append(metrics)
            self.metrics["total_duration"] += duration

            if success:
                self.metrics["success_count"] += 1
                self.metrics["results"][call_id] = result
                self.metrics["tool_results"][tool_name] = {
                    "result": result,
                    "timestamp": metrics["timestamp"],
                    "call_id": call_id,
                }
            else:
                self.metrics["error_count"] += 1

        return call_id

    def get_metrics(self) -> dict[str, Any] | None:
        async def _get_metrics():
            async with self._lock:
                return self.metrics.copy()

        try:
            # Check if we are already inside a running loop
            loop = asyncio.get_running_loop()
            if loop.is_running():
                return self.metrics.copy()
        except RuntimeError:
            # No loop is running, safe to create one or use existing
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(_get_metrics())
            finally:
                loop.close()

    def get_result(self, call_id: str) -> Any:
        """Get the result of a specific tool call by its ID."""

        async def _get_result():
            async with self._lock:
                return self.metrics["results"].get(call_id)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                future = loop.create_future()
                loop.create_task(_get_result()).add_done_callback(
                    lambda f: future.set_result(f.result())
                    if not future.done()
                    else None
                )
                return future
            return loop.run_until_complete(_get_result())
        except RuntimeError:
            return asyncio.run(_get_result())

    def get_tool_result(self, tool_name: str) -> Any:
        """Get the result of the last successful call to a specific tool."""

        async def _get_tool_result():
            async with self._lock:
                tool_result = self.metrics["tool_results"].get(tool_name)
                return tool_result["result"] if tool_result else None

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                future = loop.create_future()
                loop.create_task(_get_tool_result()).add_done_callback(
                    lambda f: future.set_result(f.result())
                    if not future.done()
                    else None
                )
                return future
            return loop.run_until_complete(_get_tool_result())
        except RuntimeError:
            return asyncio.run(_get_tool_result())

    def reset(self) -> None:
        """Reset all metrics."""

        async def _reset():
            async with self._lock:
                self.metrics = {
                    "tool_calls": [],
                    "total_duration": 0.0,
                    "success_count": 0,
                    "error_count": 0,
                    "results": {},
                    "tool_results": {},
                }
                self._prev_io = None
                self._prev_time = None

        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # Use run_coroutine_threadsafe to ensure it executes
                # even if called from a different thread.
                asyncio.run_coroutine_threadsafe(_reset(), loop)
        except RuntimeError:
            # No loop is running, safe to use asyncio.run
            asyncio.run(_reset())


# Global instance
tracker = ResourceTracker()
