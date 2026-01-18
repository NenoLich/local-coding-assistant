"""Enhanced tool management with registry, async, and streaming support.

This module provides the ToolManager class which handles tool registration,
execution, and discovery. It works with the ToolLoader for configuration
and module loading.
"""

import asyncio
import inspect
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Iterable, Iterator
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from local_coding_assistant.core.exceptions import ToolRegistryError
from local_coding_assistant.core.protocols import IToolManager
from local_coding_assistant.tools.statistics import StatisticsManager, ToolStatistics
from local_coding_assistant.tools.tool_api_generator import ToolAPIGenerator
from local_coding_assistant.tools.tool_runtime import ToolRuntime
from local_coding_assistant.tools.types import (
    ToolCategory,
    ToolExecutionMode,
    ToolExecutionRequest,
    ToolExecutionResponse,
    ToolInfo,
    ToolSource,
)

if TYPE_CHECKING:
    from local_coding_assistant.core.protocols import IConfigManager
    from local_coding_assistant.sandbox.manager import SandboxManager

from local_coding_assistant.utils.logging import get_logger

logger = get_logger("tools.tool_manager")


class ToolManager(IToolManager, Iterable[Any]):
    """Enhanced tool manager with comprehensive logging, error handling, and validation.

    This class provides:
    - Automatic tool discovery via registry
    - Async/sync execution support
    - Permission checking
    - Backward compatibility
    """

    def __init__(
        self,
        config_manager: "IConfigManager",
        sandbox_manager: "SandboxManager | None" = None,
        auto_load: bool = True,
        *,
        auto_load_registry: bool | None = None,
    ) -> None:
        """Initialize the tool manager.

        Args:
            config_manager: IConfigManager instance for tool configurations.
                If not provided, a default one will be created.
            sandbox_manager: Optional SandboxManager instance.
            auto_load: Whether to automatically load tools
            auto_load_registry: Legacy flag for tests/backwards compatibility
        """
        self._tools: dict[str, ToolInfo] = {}
        self._runtimes: dict[str, ToolRuntime] = {}
        self._sandbox_tools: dict[str, ToolRuntime] = {}
        self._statistics = StatisticsManager()
        self._config_manager: IConfigManager = config_manager
        self._sandbox_manager = sandbox_manager

        # Initialize the API generator with the sandbox's workspace if available
        tools_api_output_dir = None
        if sandbox_manager and self._config_manager.path_manager:
            tools_api_output_dir = (
                self._config_manager.path_manager.get_sandbox_guest_dir()
            )
        self._api_generator = ToolAPIGenerator(output_dir=tools_api_output_dir)

        if auto_load_registry is not None:
            auto_load = auto_load_registry

        if auto_load:
            self.load_tools()

        logger.info("ToolManager initialized with %d tools", len(self._tools))

    def _run_sync(self, awaitable: Awaitable[Any]) -> Any:
        """Safely execute an awaitable from synchronous contexts.

        Args:
            awaitable: An awaitable object to execute

        Returns:
            The result of the awaitable

        Raises:
            RuntimeError: If called from within a running event loop
        """
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is not None:
            raise RuntimeError(
                "Attempted synchronous execution while an event loop is running. "
                "Use the asynchronous interface instead."
            )

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)

            if inspect.iscoroutine(awaitable):
                coroutine = awaitable
            else:

                async def run_awaitable() -> Any:
                    return await awaitable

                coroutine = run_awaitable()

            return loop.run_until_complete(coroutine)
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    def _create_mcp_tool_executor(self, tool_info: ToolInfo) -> Any:
        """Create a dummy MCP tool executor.

        Args:
            tool_info: The MCP tool's information

        Returns:
            An instance of MCPToolExecutor

        Note:
            This is a temporary implementation that will be replaced with actual MCP client integration.
        """

        class MCPToolExecutor:
            def __init__(self, _tool_info: ToolInfo):
                self.tool_info = _tool_info
                self.endpoint = _tool_info.endpoint
                self.provider = _tool_info.provider

            async def execute(self, **kwargs) -> dict[str, Any]:
                """Execute the MCP tool with the given arguments."""
                # This is a dummy implementation that will be replaced with actual MCP client calls
                logger.info(
                    "[MCP Tool] Executing %s via %s at %s with args: %s",
                    self.tool_info.name,
                    self.provider,
                    self.endpoint,
                    kwargs,
                )

                # Simulate network delay
                await asyncio.sleep(0.5)

                # Return a dummy response
                return {
                    "status": "success",
                    "result": f"MCP tool '{self.tool_info.name}' executed successfully",
                    "provider": self.provider,
                    "endpoint": self.endpoint,
                    "arguments": kwargs,
                }

        return MCPToolExecutor(tool_info)

    def load_tools(self) -> None:
        """Load tools using ConfigManager.

        This will:
        1. Clear any existing tool instances and stats
        2. Get the latest tool configurations from ConfigManager
        3. Initialize tool instances, including MCP tools

        Raises:
            RuntimeError: If there's an error loading tools
        """
        self._tools.clear()
        self._runtimes.clear()
        self._sandbox_tools.clear()

        try:
            # Get tool configs from ConfigManager
            tool_configs = self._config_manager.get_tools()
            logger.info("Loaded %d tool configurations", len(tool_configs))

            # Initialize tool instances
            for tool_id, tool_config in tool_configs.items():
                try:
                    # Convert ToolConfig to ToolInfo for runtime use
                    if hasattr(tool_config, "to_tool_info"):
                        tool_info = tool_config.to_tool_info()
                    else:
                        raise ToolRegistryError(
                            f"Failed to load tool '{tool_id}': ToolConfig does not have a 'to_tool_info' method"
                        )

                    # Always store the tool info, even for disabled or unavailable tools
                    self._tools[tool_id] = tool_info

                    # Skip instance creation for disabled tools
                    if not tool_config.enabled:
                        logger.debug(
                            "Skipping instance creation for disabled tool: %s", tool_id
                        )
                        continue

                    # Skip instance creation for unavailable tools
                    if not tool_config.available:
                        logger.debug(
                            "Skipping instance creation for unavailable tool: %s",
                            tool_id,
                        )
                        continue

                    runtime = self._build_runtime(tool_id, tool_info)

                    # Track sandbox tools separately
                    if tool_info.source == ToolSource.SANDBOX:
                        self._sandbox_tools[tool_id] = runtime
                        logger.debug("Registered sandbox tool: %s", tool_id)

                    self._runtimes[tool_id] = runtime
                    logger.debug("Successfully initialized tool runtime: %s", tool_id)

                except Exception as e:
                    logger.error(
                        "Failed to initialize tool %s: %s",
                        tool_id,
                        str(e),
                        exc_info=True,
                    )

        except Exception as e:
            logger.error("Error loading tools: %s", str(e), exc_info=True)
            raise RuntimeError(f"Failed to load tools: {e}") from e

        # Generate tools API for sandbox tools if any exist
        if self._sandbox_tools and self._sandbox_manager:
            try:
                logger.info(
                    "Generating tools API for %d sandbox tools",
                    len(self._sandbox_tools),
                )
                self.generate_tools_api(self._sandbox_tools)
                logger.info("Successfully generated tools API for sandbox tools")
            except Exception as e:
                logger.error(
                    "Failed to generate tools API for sandbox tools: %s",
                    str(e),
                    exc_info=True,
                )
                # Don't fail the entire tool loading process if API generation fails
                # The sandbox may still work with direct tool imports

    def reload_tools(self) -> None:
        """Reload tools from configuration files.

        This will:
        1. Tell ConfigManager to reload tool configurations
        2. Re-initialize all tool instances

        Raises:
            RuntimeError: If there's an error reloading tools
        """
        try:
            # Reload tools through ConfigManager
            self._config_manager.reload_tools()

            # Re-initialize tools
            self.load_tools()

            logger.info("Successfully reloaded %d tools", len(self._tools))

        except Exception as e:
            logger.error("Error reloading tools: %s", str(e), exc_info=True)
            raise RuntimeError(f"Failed to reload tools: {e}") from e

    # Tool loading methods have been moved to ToolLoader class

    def __iter__(self) -> Iterator[tuple[str, ToolInfo]]:
        """Iterate over registered tools.

        Yields:
            Tuples of (tool_name, tool_info) for each registered tool
        """
        yield from self._tools.items()

    def __len__(self) -> int:
        """Return the number of registered tools.

        Returns:
            Number of registered tools
        """
        return len(self._tools)

    def invoke(self, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Legacy method for backward compatibility. Same as run_tool.

        Args:
            tool_name: Name of the tool to execute
            payload: Input parameters for the tool

        Returns:
            The tool's execution result

        Raises:
            ToolRegistryError: If tool execution fails
        """
        return self.run_tool(tool_name, payload)

    def execute(self, request: ToolExecutionRequest) -> ToolExecutionResponse:
        """Execute a tool using a ToolExecutionRequest and return ToolExecutionResponse.

        Args:
            request: ToolExecutionRequest containing tool name and payload.

        Returns:
            ToolExecutionResponse with execution results.
        """
        tool_name = request.tool_name
        start_time = time.perf_counter()

        try:
            runtime = self._get_runtime(tool_name)
            result = self._run_sync(runtime.execute(request.payload))
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            self._run_sync(self._record_success(tool_name, execution_time_ms / 1000.0))

            # Handle sandbox tool responses
            if hasattr(runtime.instance, "sandbox_manager") and isinstance(
                result, dict
            ):
                # Check for final answer from sandbox
                if "final_answer" in result and result["final_answer"] is not None:
                    final_answer = result["final_answer"]
                    return ToolExecutionResponse(
                        success=True,
                        tool_name="final_answer",
                        result=final_answer.get("answer"),
                        format=final_answer.get("format", "text"),
                        metadata=final_answer.get("metadata", {}),
                        is_final=True,
                        execution_time_ms=execution_time_ms,
                        stdout=result.get("stdout"),
                        stderr=result.get("stderr"),
                    )

                # Handle regular sandbox responses
                error_message = result.get("error_message", result.get("error"))
                return ToolExecutionResponse(
                    success=result.get("success", True),
                    tool_name=tool_name,
                    result=result.get("result"),
                    error_message=error_message,
                    execution_time_ms=execution_time_ms,
                    stdout=result.get("stdout"),
                    stderr=result.get("stderr"),
                    files_created=result.get("files_created"),
                    files_modified=result.get("files_modified"),
                )

            # Handle standard tool responses
            return ToolExecutionResponse(
                success=True,
                tool_name=tool_name,
                result=result,
                execution_time_ms=execution_time_ms,
            )
        except Exception as exc:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            self._run_sync(
                self._record_error(tool_name, execution_time_ms / 1000.0, exc)
            )
            error_msg = f"Error executing tool '{tool_name}': {exc}"
            logger.exception(error_msg)

            return ToolExecutionResponse(
                success=False,
                tool_name=tool_name,
                error_message=error_msg,
                execution_time_ms=execution_time_ms,
            )

    async def execute_async(
        self, request: ToolExecutionRequest
    ) -> ToolExecutionResponse:
        """Asynchronously execute a tool using a ToolExecutionRequest and return ToolExecutionResponse.

        Args:
            request: ToolExecutionRequest containing tool name and payload.

        Returns:
            ToolExecutionResponse with execution results.
        """
        start_time = time.perf_counter()
        tool_name = request.tool_name
        payload = request.payload or {}

        try:
            # Get the tool runtime
            runtime = self._get_runtime(tool_name)
            result = await runtime.execute(payload)
            execution_time_ms = (time.perf_counter() - start_time) * 1000

            # Handle sandbox responses
            if result and result.get("response"):
                result = result.get("response")
                # Extract tool_calls and system_metrics from result
                tool_calls = (
                    result.get("tool_calls") if result.get("tool_calls") else None
                )
                system_metrics = (
                    result.get("system_metrics")
                    if result.get("system_metrics")
                    else None
                )

                # If no tool_calls, but we have a result, create a synthetic tool call
                if not tool_calls and result.get("result"):
                    tool_calls = [
                        {
                            "tool_name": tool_name,
                            "result": result.get("result"),
                            "start_time": datetime.now(UTC)
                            - timedelta(seconds=execution_time_ms / 1000.0),
                            "end_time": datetime.now(UTC),
                            "success": True,
                            "resource_metrics": system_metrics or [],
                        }
                    ]

                await self._record_success(
                    tool_name,
                    execution_time_ms / 1000.0,
                    tool_calls=tool_calls,
                    system_metrics=system_metrics,
                )

                # Check for final answer from sandbox
                if result.get("final_answer"):
                    final_answer = result.get("final_answer")
                    return ToolExecutionResponse(
                        success=True,
                        tool_name="final_answer",
                        result=final_answer.get("answer"),
                        format=final_answer.get("format", "text"),
                        metadata=final_answer.get("metadata", {}),
                        is_final=True,
                        execution_time_ms=execution_time_ms,
                        stdout=result.get("stdout"),
                        stderr=result.get("stderr"),
                    )

                # Handle regular sandbox responses
                return ToolExecutionResponse(
                    success=result.get("success"),
                    tool_name=tool_name,
                    result=result.get("result"),
                    error_message=result.get("error"),
                    execution_time_ms=execution_time_ms,
                    stdout=result.get("stdout"),
                    stderr=result.get("stderr"),
                    files_created=result.get("files_created"),
                    files_modified=result.get("files_modified"),
                )

            else:
                # Handle non-SandboxExecutionResponse results (legacy)
                await self._record_success(tool_name, execution_time_ms / 1000.0)

                return ToolExecutionResponse(
                    success=True,
                    tool_name=tool_name,
                    result=result,
                    execution_time_ms=execution_time_ms,
                )

        except Exception as exc:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            # Get resource_usage from exception if it's a SandboxExecutionError
            resource_usage = (
                getattr(exc, "resource_usage", None)
                if hasattr(exc, "resource_usage")
                else None
            )
            await self._record_error(
                tool_name,
                execution_time_ms / 1000.0,
                exc,
                resource_usage=resource_usage,
            )
            error_msg = f"Error executing tool '{tool_name}': {exc}"
            logger.exception(error_msg)

            return ToolExecutionResponse(
                success=False,
                tool_name=tool_name,
                error_message=error_msg,
                execution_time_ms=execution_time_ms,
            )

    def get_tool_info(self, tool_name: str) -> ToolInfo | None:
        """Get detailed information about a registered tool.

        Args:
            tool_name: Name of the tool to get info for.

        Returns:
            ToolInfo object with tool details, or None if not found.
        """
        return self._tools.get(tool_name)

    def list_tools(
        self,
        available_only: bool = True,
        execution_mode: str | ToolExecutionMode | None = None,
        category: str | ToolCategory | None = None,
    ) -> list[ToolInfo]:
        """List all registered tools, optionally filtered by execution mode and/or category.

        Args:
            available_only: If True, only returns tools that are available.
            execution_mode: Optional execution mode to filter tools by.
            category: Optional category to filter tools by (can be string or ToolCategory).
                     If the category doesn't exist, returns an empty list.

        Returns:
            List of ToolInfo objects.
        """
        filtered_tools = list(self._tools.values())
        if available_only:
            filtered_tools = [tool for tool in filtered_tools if tool.available]

        if execution_mode:
            filtered_tools = [
                tool for tool in filtered_tools if tool.execution_mode == execution_mode
            ]

        if category is None:
            return filtered_tools

        # Convert string category to ToolCategory if needed
        if isinstance(category, str):
            try:
                category = ToolCategory(category)
            except ValueError:
                # For unknown categories, return an empty list
                return []

        return [info for info in filtered_tools if info.category == category]

    def get_execution_stats(self, tool_name: str | None = None) -> dict:
        """Get execution statistics for tools.

        Args:
            tool_name: Optional name of the tool to get stats for. If None, returns all stats.

        Returns:
            Dictionary containing execution statistics.
        """
        if tool_name:
            tool_stats = self._statistics.get_tool_stats(tool_name)
            if tool_stats and isinstance(tool_stats, ToolStatistics):
                return {
                    "total_executions": tool_stats.total_executions,
                    "success_count": tool_stats.success_count,
                    "error_count": tool_stats.error_count,
                    "success_rate": tool_stats.success_rate,
                    "avg_duration": tool_stats.avg_duration,
                    "first_execution": tool_stats.first_execution,
                    "last_execution": tool_stats.last_execution,
                    "metrics_summary": tool_stats.get_metrics_summary(),
                }
            return {}

        # Return all tools' stats
        result = {}
        for name in self._tools:
            tool_stats = self._statistics.get_tool_stats(name)
            if tool_stats and isinstance(tool_stats, ToolStatistics):
                result[name] = {
                    "total_executions": tool_stats.total_executions,
                    "success_count": tool_stats.success_count,
                    "error_count": tool_stats.error_count,
                    "success_rate": tool_stats.success_rate,
                    "avg_duration": tool_stats.avg_duration,
                    "first_execution": tool_stats.first_execution,
                    "last_execution": tool_stats.last_execution,
                    "metrics_summary": tool_stats.get_metrics_summary(),
                }
        return result

    def get_metric_history(
        self,
        metric_name: str,
        tool_name: str | None = None,
        time_range: tuple[datetime, datetime] | None = None,
    ) -> list[dict]:
        """Get historical data for a specific metric.

        Args:
            metric_name: Name of the metric to retrieve
            tool_name: Optional tool name to filter by
            time_range: Optional (start, end) datetime range to filter by

        Returns:
            List of metric values with timestamps
        """
        return self._statistics.get_metric_history(metric_name, tool_name, time_range)

    def get_system_stats(self) -> dict:
        """Get system-wide statistics.

        Returns:
            Dictionary containing system statistics
        """
        stats = self._statistics.get_system_stats()
        return {
            "total_executions": stats.total_executions,
            "total_duration": stats.total_duration,
            "avg_duration": stats.avg_duration,
            "first_execution": stats.first_execution,
            "last_execution": stats.last_execution,
            "metrics_summary": stats.get_metrics_summary(),
        }

    def run_tool(
        self,
        tool_name: str,
        payload: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Run a tool with the given parameters.

        Args:
            tool_name: Name of the tool to run
            payload: Payload to pass to the tool
            **kwargs: Additional keyword arguments

        Returns:
            The result of the tool execution

        Raises:
            ToolRegistryError: If the tool is not found, not enabled, not available, or execution fails
        """
        return self._run_sync(self.run_tool_async(tool_name, payload, **kwargs))

    async def run_tool_async(
        self,
        tool_name: str,
        parameters: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Run a tool asynchronously with the given parameters.

        Args:
            tool_name: Name of the tool to run
            parameters: Parameters to pass to the tool
            **kwargs: Additional keyword arguments

        Returns:
            The result of the tool execution

        Raises:
            ToolRegistryError: If the tool is not found, not enabled, not available, or execution fails
        """
        runtime = self._get_runtime(tool_name)

        start_time = time.perf_counter()
        try:
            result = await runtime.execute(parameters)
            execution_time = time.perf_counter() - start_time
            await self._record_success(tool_name, execution_time)
            return result
        except Exception as exc:
            execution_time = time.perf_counter() - start_time
            await self._record_error(tool_name, execution_time, exc)
            raise ToolRegistryError(
                f"Error executing tool '{tool_name}': {exc}"
            ) from exc

    async def _record_success(
        self,
        tool_name: str,
        duration_sec: float,
        tool_calls: list[dict] | None = None,
        system_metrics: list[dict] | None = None,
    ) -> None:
        """Record a successful tool execution with detailed metrics.

        Args:
            tool_name: Name of the tool that was executed
            duration_sec: Total execution time in seconds
            tool_calls: List of tool calls with their individual metrics
            system_metrics: List of system-level metrics from the sandbox
        """
        # Log basic execution info
        logger.debug(
            "Tool '%s' executed successfully in %.2fs", tool_name, duration_sec
        )

        # Record system metrics if available
        if system_metrics:
            logger.debug("Recording %d system metrics", len(system_metrics))
            await self._statistics.record_system_metrics(system_metrics, duration_sec)

        # Record tool calls if available
        if tool_calls:
            logger.debug("Recording %d tool calls", len(tool_calls))
            for call in tool_calls:
                try:
                    # Create a unique call ID by hashing a string representation of the call
                    call_id = call.get("call_id")
                    if not call_id:
                        # Create a stable string representation of the call
                        call_repr = f"{call.get('tool_name', tool_name)}:{call.get('start_time')}:{call.get('end_time')}"
                        call_id = str(hash(call_repr))

                    await self._statistics.record_tool_call(
                        tool_name=call.get("tool_name", tool_name),
                        call_id=call_id,
                        start_time=call.get("start_time", datetime.now(UTC)),
                        end_time=call.get("end_time", datetime.now(UTC)),
                        success=call.get("success", True),
                        error=call.get("error"),
                        resource_metrics=call.get("resource_metrics", []),
                        metadata=call.get("metadata", {}),
                    )
                except Exception as e:
                    logger.error(
                        "Failed to record tool call metrics",
                        error=str(e),
                        exc_info=True,
                    )
        else:
            # If no tool calls, record the main tool execution
            await self._statistics.record_tool_call(
                tool_name=tool_name,
                call_id=str(hash((tool_name, time.time()))),
                start_time=datetime.now(UTC) - timedelta(seconds=duration_sec),
                end_time=datetime.now(UTC),
                success=True,
                resource_metrics=system_metrics or [],
            )

    async def _record_error(
        self,
        tool_name: str,
        duration_sec: float,
        error: Exception,
        resource_usage: dict | None = None,
    ) -> None:
        logger.error(
            "Error in tool '%s': %s",
            tool_name,
            str(error),
            exc_info=logger.is_enabled_for(logging.DEBUG)
            if hasattr(logger, "is_enabled_for")
            else True,
        )

        # Record the error in the statistics system
        await self._statistics.record_tool_call(
            tool_name=tool_name,
            call_id=str(hash((tool_name, time.time()))),
            start_time=datetime.now(UTC) - timedelta(seconds=duration_sec),
            end_time=datetime.now(UTC),
            success=False,
            error=str(error),
            resource_metrics=[resource_usage]
            if resource_usage and isinstance(resource_usage, dict)
            else [],
        )

    def _build_runtime(self, tool_id: str, tool_info: "ToolInfo") -> "ToolRuntime":
        """Build a ToolRuntime instance from a tool configuration.

        Args:
            tool_id: Unique identifier for the tool
            tool_info: Tool information

        Returns:
            A configured ToolRuntime instance

        Raises:
            ToolRegistryError: If there's an error building the runtime
        """
        try:
            # Determine the tool kind based on source
            if tool_info.source == ToolSource.MCP:
                tool_instance = self._create_mcp_tool_executor(tool_info)
                kind = "mcp"
            else:
                # The tool class should already be imported by ToolLoader
                if not hasattr(tool_info, "tool_class") or tool_info.tool_class is None:
                    raise ToolRegistryError(
                        f"Tool '{tool_id}' is missing tool class. "
                        "Ensure ToolLoader is properly loading the tool class."
                    )

                try:
                    tool_instance = tool_info.tool_class()
                    kind = "tool"
                except Exception as e:
                    logger.error(
                        "Failed to instantiate tool class for '%s': %s",
                        tool_id,
                        str(e),
                        exc_info=logger.is_enabled_for(logging.DEBUG)
                        if hasattr(logger, "is_enabled_for")
                        else True,
                    )
                    raise ToolRegistryError(
                        f"Failed to instantiate tool '{tool_id}': {e!s}"
                    ) from e

            # Use the pre-computed values from config
            run_is_async = tool_info.is_async
            supports_streaming = tool_info.supports_streaming

            # Inject sandbox_manager into tools that need it
            if hasattr(tool_instance, "sandbox_manager") and self._sandbox_manager:
                tool_instance.sandbox_manager = self._sandbox_manager

            # Check for input/output validation models
            has_input_validation = (
                hasattr(tool_instance, "Input")
                and issubclass(tool_instance.Input, BaseModel)
                if hasattr(tool_instance, "Input")
                else False
            )

            has_output_validation = (
                hasattr(tool_instance, "Output")
                and issubclass(tool_instance.Output, BaseModel)
                if hasattr(tool_instance, "Output")
                else False
            )

            # Create the runtime
            tool_runtime = ToolRuntime(
                info=tool_info,
                instance=tool_instance,
                kind=kind,
                run_is_async=run_is_async,
                supports_streaming=supports_streaming,
                has_input_validation=has_input_validation,
                has_output_validation=has_output_validation,
            )

            # Return the runtime
            return tool_runtime

        except Exception as e:
            logger.error(
                "Error building runtime for tool '%s'",
                tool_id,
                error=str(e),
                exc_info=True,
            )
            raise ToolRegistryError(
                f"Failed to build runtime for tool '{tool_id}': {e!s}"
            ) from e

    def _runtime_from_instance(
        self, tool_info: "ToolInfo", tool_instance: Any
    ) -> "ToolRuntime":
        """Create a ToolRuntime from an existing tool instance.

        Args:
            tool_info: Tool information
            tool_instance: The tool instance

        Returns:
            A configured ToolRuntime instance
        """
        # Determine if the tool has async run method
        run_method = getattr(tool_instance, "run", None)
        run_is_async = run_method is not None and inspect.iscoroutinefunction(
            run_method
        )

        # Check if tool supports streaming
        stream_method = getattr(tool_instance, "stream", None)
        supports_streaming = stream_method is not None and (
            inspect.iscoroutinefunction(stream_method)
            or inspect.isasyncgenfunction(stream_method)
        )

        # Check for input/output validation models
        has_input_validation = (
            hasattr(tool_instance, "Input")
            and issubclass(tool_instance.Input, BaseModel)
            if hasattr(tool_instance, "Input")
            else False
        )

        has_output_validation = (
            hasattr(tool_instance, "Output")
            and issubclass(tool_instance.Output, BaseModel)
            if hasattr(tool_instance, "Output")
            else False
        )

        # Determine the kind of tool
        kind = "mcp" if tool_info.source == ToolSource.MCP else "tool"

        return ToolRuntime(
            info=tool_info,
            instance=tool_instance,
            kind=kind,
            run_is_async=run_is_async,
            supports_streaming=supports_streaming,
            has_input_validation=has_input_validation,
            has_output_validation=has_output_validation,
        )

    def get_sandbox_tools_prompt(self) -> str:
        """Generate prompt segment for non-sandbox tools in PTC mode.

        Returns:
            Formatted string containing tool documentation and usage examples
            for non-sandbox tools, or an empty string if no tools found.
        """
        prompt_segments = []

        for tool_name, tool_info in self._tools.items():
            # Skip none-sandbox tools
            if tool_info.source != ToolSource.SANDBOX:
                continue

            if not tool_info.available:
                continue

            # Use tool_info.description as the main description
            description = tool_info.description or "No description available."

            # Generate parameter documentation
            parameters = tool_info.parameters.get("properties", {})
            required_params = tool_info.parameters.get("required", [])

            # Format parameters section
            params_doc = []
            for param_name, param_schema in parameters.items():
                param_type = param_schema.get("type", "Any")
                param_desc = param_schema.get(
                    "description", "No description available."
                )
                is_required = param_name in required_params
                req_text = " (required)" if is_required else " (optional)"
                params_doc.append(
                    f"  - {param_name}: {param_desc} [Type: {param_type}]{req_text}"
                )

            # Format the tool documentation
            tool_doc = f"""
## {tool_name}

{description}

### Parameters:
{"\n".join(params_doc) if params_doc else "No parameters."}

### Usage Example:
```python
from tools_api import {tool_name}
"""

            # Generate example with individual parameters instead of a dictionary
            if parameters:
                # Create a list of parameter assignments
                param_assignments = []
                for param_name, param_schema in parameters.items():
                    param_type = param_schema.get("type", "Any")
                    if isinstance(param_type, list):
                        param_type = param_type[0] if param_type else "Any"
                    param_assignments.append(
                        f"{param_name}=<{param_name}: {param_type}>"
                    )

                # Add the function call with individual parameters
                tool_doc += f"\nresult = {tool_name}({', '.join(param_assignments)})"
            else:
                tool_doc += f"\nresult = {tool_name}()"

            # Add the rest of the example
            tool_doc += """
print(result)
```"""

            prompt_segments.append(tool_doc.strip())

        return "\n\n".join(prompt_segments) if prompt_segments else ""

    async def _build_sandbox_result_string(
        self, sandbox_result: dict, tool_name: str, session_id: str
    ) -> str:
        """Build a detailed result string from sandbox execution.

        Args:
            sandbox_result: The result dictionary from sandbox execution
            tool_name: Name of the executed tool
            session_id: ID of the sandbox session

        Returns:
            Formatted string with execution details
        """
        result = [
            "=== Sandbox Execution Details ===",
            f"Tool: {tool_name}",
            f"Session: {session_id}",
            f"Duration: {sandbox_result.get('duration', 0):.2f}s",
            f"Exit Code: {sandbox_result.get('return_code', 'N/A')}",
        ]

        # Add the actual result if present
        if sandbox_result.get("result"):
            result.extend(["\n--- Result ---", str(sandbox_result.get("result"))])

        # Add stdout if present
        if sandbox_result.get("stdout"):
            result.extend(
                ["\n--- Standard Output ---", sandbox_result.get("stdout", "").strip()]
            )

        # Add stderr if present
        if sandbox_result.get("stderr"):
            result.extend(
                ["\n--- Standard Error ---", sandbox_result.get("stderr", "").strip()]
            )

        # Add file operations if present
        files_created = sandbox_result.get("files_created")
        if files_created and isinstance(files_created, list):
            result.append("\n--- Files Created ---")
            result.extend(f"- {f}" for f in files_created)

        files_modified = sandbox_result.get("files_modified")
        if files_modified and isinstance(files_modified, list):
            result.append("\n--- Files Modified ---")
            result.extend(f"- {f}" for f in files_modified)

        return "\n".join(result)

    async def execute_tool_in_sandbox(
        self, tool_name: str, payload: dict[str, Any], session_id: str = "default"
    ) -> str:
        """Execute a tool in a sandboxed environment.

        Args:
            tool_name: Name of the tool to execute
            payload: Input parameters for the tool
            session_id: Session ID for the sandbox

        Returns:
            A formatted string with all sandbox execution details

        Raises:
            ToolRegistryError: If sandbox execution fails or sandbox is not available
        """
        start_time = time.perf_counter()

        # Get the tool's runtime
        runtime = self._get_runtime("execute_python_code")

        # Generate code that imports and calls the tool
        tool_call_code = await self._generate_tool_call_code(tool_name, payload)

        if (
            not runtime.instance
            or not hasattr(runtime.instance, "sandbox_manager")
            or not runtime.instance.sandbox_manager
        ):
            error_msg = "Sandbox execution is not available. Make sure the sandbox tools are registered."
            execution_time = (time.perf_counter() - start_time) * 1000
            await self._record_error(
                "execute_python_code", execution_time / 1000.0, Exception(error_msg)
            )
            raise ToolRegistryError(error_msg)

        # Execute the code in the sandbox
        try:
            # The ExecutePythonCodeTool returns a dict with success, result, stdout, stderr, error
            sandbox_result = await runtime.execute(
                {"code": tool_call_code, "session_id": session_id}
            )
            sandbox_result = sandbox_result.get("response")
            logger.debug("Sandbox result received", sandbox_result=sandbox_result)

            execution_time = (time.perf_counter() - start_time) * 1000  # in ms

            # If the sandbox execution itself failed
            if not isinstance(sandbox_result, dict) or not sandbox_result.get(
                "success"
            ):
                error = (
                    sandbox_result.get("error", "Unknown error in sandbox execution")
                    if isinstance(sandbox_result, dict)
                    else "Invalid response from sandbox"
                )
                error_msg = f"Sandbox execution failed: {error}"
                await self._record_error(
                    "execute_python_code", execution_time / 1000.0, Exception(error_msg)
                )
                raise ToolRegistryError(f"Sandbox execution failed: {error_msg}")

            # Extract resource metrics
            resource_metrics = []
            if sandbox_result.get("tool_calls"):
                tool_call = next(iter(sandbox_result.get("tool_calls")))
                if tool_call.get("resource_metrics"):
                    resource_metrics = tool_call.get("resource_metrics")

            # Record successful execution
            await self._record_success(
                tool_name,
                execution_time / 1000.0,  # Convert to seconds
                tool_calls=[
                    {
                        "tool_name": tool_name,
                        "result": sandbox_result.get("result"),
                        "start_time": sandbox_result.get("start_time"),
                        "end_time": sandbox_result.get("end_time") or datetime.now(UTC),
                        "success": True,
                        "resource_metrics": resource_metrics,
                    }
                ],
                system_metrics=sandbox_result.get("system_metrics"),
            )

            # Build and return the result string
            return await self._build_sandbox_result_string(
                sandbox_result, tool_name, session_id
            )

        except ToolRegistryError as e:
            # Re-raise ToolRegistryError as is, but record the error first
            execution_time = (time.perf_counter() - start_time) * 1000
            await self._record_error(tool_name, execution_time / 1000.0, e)
            raise

        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            await self._record_error(tool_name, execution_time / 1000.0, e)
            raise ToolRegistryError(f"Error during sandbox execution: {e!s}") from e

    async def _generate_tool_call_code(
        self, tool_name: str, payload: dict[str, Any]
    ) -> str:
        """Generate Python code to call a tool with the given payload.

        Args:
            tool_name: Name of the tool to call
            payload: Arguments to pass to the tool

        Returns:
            Python code as a string
        """
        # Convert payload to a properly formatted dictionary string
        args_str = ", ".join(f"{k}={v!r}" for k, v in payload.items())

        tool_call_code = f"""
from tools_api import {tool_name}

# Call the tool with the provided arguments
result = {tool_name}({args_str})

print(result)
        """

        return tool_call_code

    def _get_runtime(self, tool_name: str) -> ToolRuntime:
        """Get the runtime instance for a tool.

        Args:
            tool_name: Name of the tool to get runtime for

        Returns:
            The ToolRuntime instance for the tool

        Raises:
            ToolRegistryError: If the tool is not found or not available
        """
        if tool_name not in self._runtimes:
            raise ToolRegistryError(f"Tool '{tool_name}' not found")

        runtime = self._runtimes[tool_name]

        # Check if tool is enabled and available
        tool_info = self._tools.get(tool_name)
        if tool_info and not tool_info.enabled:
            raise ToolRegistryError(f"Tool '{tool_name}' is disabled")

        if tool_info and not tool_info.available:
            raise ToolRegistryError(f"Tool '{tool_name}' is not available")

        return runtime

    def has_runtime(self, tool_name: str) -> bool:
        """Check if a tool with the given name has a valid runtime.

        Args:
            tool_name: Name of the tool to check

        Returns:
            bool: True if the tool exists and has a valid runtime, False otherwise
        """
        return tool_name in self._runtimes and self._runtimes[tool_name] is not None

    def get_tool(self, tool_name: str) -> Any:
        """Get a tool instance by name."""
        runtime = self._get_runtime(tool_name)
        return runtime.instance

    async def arun_tool(
        self, tool_name: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Asynchronously execute a tool.

        Args:
            tool_name: Name of the tool to execute
            payload: Input parameters for the tool

        Returns:
            Tool execution result

        Raises:
            ToolRegistryError: If tool execution fails or tool is not async
        """

        response = await self.execute_async(
            ToolExecutionRequest(tool_name=tool_name, payload=payload)
        )

        return response.model_dump()

    async def stream_tool(
        self, tool_name: str, payload: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream results from a tool that supports streaming.

        Args:
            tool_name: Name of the tool to execute
            payload: Input parameters for the tool

        Yields:
            Chunks of tool output

        Raises:
            ToolRegistryError: If tool execution fails or tool doesn't support streaming
        """
        start_time = time.time()

        try:
            runtime = self._get_runtime(tool_name)

            if not runtime.supports_streaming:
                raise ToolRegistryError(
                    f"Tool '{tool_name}' does not support streaming"
                )

            if not runtime.run_is_async:
                raise ToolRegistryError(f"Streaming tool '{tool_name}' must be async")

            logger.info("Starting stream from tool: %s", tool_name)

            # Stream chunks using the runtime
            async for chunk in runtime.stream(payload):
                yield chunk

            # Record success with execution time in ms
            execution_time = (time.time() - start_time) * 1000
            await self._record_success(tool_name, execution_time)
            logger.info(
                "Streaming tool '%s' completed in %.2fms", tool_name, execution_time
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            await self._record_error(tool_name, execution_time, e)
            error_msg = f"Error streaming from tool '{tool_name}': {e!s}"
            logger.exception(error_msg)
            raise ToolRegistryError(error_msg) from e

    async def run_programmatic_tool_call(
        self,
        code: str,
        session_id: str | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> ToolExecutionResponse:
        """Execute code using the programmatic tool calling interface (sandbox).

        Args:
            code: The Python code to execute.
            session_id: The session ID for context persistence.
            env_vars: Optional environment variables.

        Returns:
            The execution response.
        """
        if not self._sandbox_manager:
            return ToolExecutionResponse(
                tool_name="execute_python_code",
                success=False,
                error_message="Sandbox manager is not initialized",
                result="",
                execution_time_ms=0,
            )

        if not session_id:
            session_id = self._config_manager.global_config.sandbox.session_id

        request = ToolExecutionRequest(
            tool_name="execute_python_code",
            payload={"code": code, "session_id": session_id, "env_vars": env_vars},
        )

        return await self.execute_async(request)

    def generate_tools_api(
        self, tools: dict[str, "ToolRuntime"], output_dir: str | None = None
    ) -> str:
        """Generate the tools API module.

        Args:
            tools: Dictionary of tools to generate the API for.
            output_dir: Directory to generate the API in. If None, uses the sandbox workspace.

        Returns:
            Path to the generated API directory

        Note:
            If sandbox is enabled, the tools will be available at /app/tools in the container.
            The directory structure will be:
            - /app/tools/tools_api/__init__.py
            - /app/tools/tools_api/tools_api.py
        """
        return self._api_generator.generate(tools, output_dir)

    def cleanup(self) -> None:
        """Clean up any resources used by the tool manager."""
        if hasattr(self, "_api_generator"):
            self._api_generator.cleanup()

        # Clear all tool collections
        self._tools.clear()
        self._runtimes.clear()
        self._sandbox_tools.clear()

    # Update __del__ to call cleanup
    def __del__(self):
        """Clean up when the tool manager is destroyed."""
        self.cleanup()
