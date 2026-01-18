from pathlib import Path
from typing import TYPE_CHECKING

from local_coding_assistant.core.exceptions import ToolRegistryError
from local_coding_assistant.tools.types import ToolInfo
from local_coding_assistant.utils.logging import get_logger

if TYPE_CHECKING:
    from local_coding_assistant.tools.tool_manager import ToolRuntime

logger = get_logger("tools.tool_api_generator")


class ToolAPIGenerator:
    """Generates API stubs for tools to be used in sandbox environments."""

    def __init__(self, output_dir: str | Path | None = None):
        """Initialize the ToolAPIGenerator.

        Args:
            output_dir: Directory to store generated API files. If None, must be provided
                     when calling generate(). Can be either a string or Path object.
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self._generated_files = set()

    def generate(
        self, tools: dict[str, "ToolRuntime"], output_dir: str | Path | None = None
    ) -> str:
        """Generate the tools API module.

        Args:
            tools: Dictionary of tool names to ToolRuntime instances
            output_dir: Directory to generate the API in. If None, uses the one from __init__.

        Returns:
            Path to the generated API directory

        Raises:
            ValueError: If no output directory is provided and none was set in __init__
            ToolRegistryError: If API generation fails
        """
        try:
            if not tools:
                raise ValueError("No tools provided for API generation")

            # Convert output_dir to Path if it's a string
            if output_dir is not None and not isinstance(output_dir, Path):
                output_dir = Path(output_dir)
            output_dir = output_dir or self.output_dir

            if output_dir is None:
                raise ValueError(
                    "Output directory must be provided either in constructor or method call"
                )

            # Create the output directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate the tools module
            output_path = output_dir / "tools_api.py"
            self._generate_tools_module(output_path, tools)

            return str(output_path.resolve())

        except Exception as e:
            raise ToolRegistryError(f"Failed to generate tools API: {e!s}") from e

    def _generate_tools_module(
        self, output_path: Path, tools: dict[str, "ToolRuntime"]
    ) -> None:
        """Generate the tools_api.py module with tool stubs.

        Args:
            output_path: Path to the output file
            tools: Dictionary of tool names to ToolRuntime instances
        """
        code = [
            "# type: ignore\n",
            "from typing import Any, Dict, Optional, Union, List, AsyncIterator\n",
            "import asyncio\n",
            "from resource_tracker import tracker\n\n",
            "class ToolError(Exception):\n",
            '    """Base exception for tool execution errors."""\n',
            "    pass\n\n",
        ]

        # Add tool imports
        if tools:
            code.append("# Tool imports\n")
            code.append("import sys\n")
            code.append("import os\n")
            code.append("# Add /tools to Python path for imports\n")
            code.append("if '/tools' not in sys.path:\n")
            code.append("    sys.path.insert(0, '/tools')\n\n")

            imported_classes = set()

            for _tool_name, runtime in tools.items():
                if runtime.info.tool_class is None:
                    continue

                class_name = runtime.info.tool_class.__name__
                if class_name not in imported_classes:
                    # Import directly from tools.* since we mounted the tools directory in the workspace
                    code.append(f"from sandbox_tools import {class_name}\n")
                    imported_classes.add(class_name)
            code.append("\n")

        # Add tool functions
        for tool_name, runtime in tools.items():
            code.extend(self._generate_tool_function(tool_name, runtime))

        # Write the module (overwrite if exists)
        output_path.write_text("".join(code), encoding="utf-8")

    def _generate_tool_function(
        self, tool_name: str, runtime: "ToolRuntime"
    ) -> list[str]:
        """Generate code for both sync and async versions of a tool function with resource tracking."""
        tool_info = runtime.info
        if tool_info.tool_class is None:
            return []

        is_async = runtime.run_is_async
        class_name = tool_info.tool_class.__name__

        # Get parameters
        params = self._get_parameters(tool_info)
        param_names = [p.split(":")[0].strip() for p in params]
        param_str = ", ".join(params)

        code = []

        # Generate async version (the main implementation)
        code.extend(
            [
                f"\nasync def {tool_name}_async({param_str}) -> Any:\n",
                f'    """Async version of {tool_name} function with resource tracking."""\n',
                f"    tool = {class_name}()\n",
                "    run_kwargs = {}\n",
            ]
        )

        # Add parameter assignments
        for param in param_names:
            code.extend(
                [
                    f"    if '{param}' in locals() and {param} is not None:\n",
                    f"        run_kwargs['{param}'] = {param}\n",
                ]
            )

        # Add the actual execution with tracking
        code.extend(
            [
                f'    @tracker.track(tool_name="{tool_name}")\n',
                "    async def _execute():\n",
                "        try:\n",
            ]
        )

        if is_async:
            code.append("            return await tool.run(**run_kwargs)\n")
        else:
            code.append("            return tool.run(**run_kwargs)\n")

        code.extend(
            [
                "        except Exception as e:\n",
                f'            raise ToolError(f"Error in {tool_name}: {{str(e)}}") from e\n\n',
                "    return await _execute()\n",
            ]
        )

        # Sync version simply calls the async version
        code.extend(
            [
                f"\ndef {tool_name}({param_str}) -> Any:\n",
                f'    """Synchronous version of {tool_name} function."""\n',
                "    loop = None\n",
                "    try:\n",
                "        loop = asyncio.new_event_loop()\n",
                "        asyncio.set_event_loop(loop)\n",
                f"        return loop.run_until_complete({tool_name}_async({', '.join(param_names)}))\n",
                "    except Exception as e:\n",
                f'        raise ToolError(f"Error in sync wrapper for {tool_name}: {{str(e)}}") from e\n',
                "    finally:\n",
                "        loop.close()\n",
            ]
        )

        # Add streaming support if available
        if runtime.supports_streaming:
            code.extend(
                self._generate_stream_function(tool_name, class_name, param_names)
            )

        return code

    def _generate_stream_function(
        self, tool_name: str, class_name: str, param_names: list[str]
    ) -> list[str]:
        """Generate streaming functions for a tool (both async and sync versions).

        Args:
            tool_name: Name of the tool
            class_name: Name of the tool's class
            param_names: List of parameter names for the tool

        Returns:
            List of strings representing the lines of the generated functions
        """
        param_str = ", ".join([p for p in param_names if p != "self"])

        code = [
            f"\nasync def stream_{tool_name}_async({param_str}) -> AsyncIterator[Any]:\n",
            f'    """Async version: Stream results from {tool_name}."""\n',
            f"    tool = {class_name}()\n",
            "    if not hasattr(tool, 'stream'):\n",
            f"        raise ToolError('Tool \"{tool_name}\" does not support streaming')\n",
            "    try:\n",
            f"        async for chunk in tool.stream({param_str}):\n",
            "            yield chunk\n",
            "    except Exception as e:\n",
            f'        raise ToolError(f"Error in stream_{tool_name}: {{str(e)}}") from e\n',
            "\n",
            "# Sync version of the streaming function\n",
            f"def stream_{tool_name}({param_str}) -> AsyncIterator[Any]:\n",
            f'    """Synchronous version: Stream results from {tool_name}."""\n',
            "    try:\n",
            "        # Create a new event loop for this thread\n",
            "        loop = asyncio.new_event_loop()\n",
            "        asyncio.set_event_loop(loop)\n",
            "        try:\n",
            "            async def _async_gen() -> AsyncIterator[Any]:\n",
            f"                async for chunk in stream_{tool_name}_async({param_str}):\n",
            "                    yield chunk\n",
            "            \n",
            "            # Create a queue to collect results\n",
            "            result_queue = asyncio.Queue()\n",
            "            \n",
            "            # Run the async generator in a task\n",
            "            async def _collect() -> None:\n",
            "                try:\n",
            "                    async for chunk in _async_gen():\n",
            "                        await result_queue.put(('data', chunk))\n",
            "                    await result_queue.put(('done', None))\n",
            "                except Exception as e:\n",
            "                    await result_queue.put(('error', e))\n",
            "            \n",
            "            # Start the collection task\n",
            "            task = loop.create_task(_collect())\n",
            "            \n",
            "            # Yield results as they come in\n",
            "            while True:\n",
            "                try:\n",
            "                    msg, value = loop.run_until_complete(result_queue.get())\n",
            "                    if msg == 'data':\n",
            "                        yield value\n",
            "                    elif msg == 'done':\n",
            "                        break\n",
            "                    elif msg == 'error':\n",
            "                        raise value\n",
            "                except Exception as e:\n",
            "                    task.cancel()\n",
            "                    raise e\n",
            "        finally:\n",
            "            # Clean up the event loop\n",
            "            loop.close()\n",
            "    except Exception as e:\n",
            f'        raise ToolError(f"Error in sync stream_{tool_name}: {{str(e)}}") from e\n',
            "\n",
        ]

        return code

    @staticmethod
    def _get_parameters(tool_info: "ToolInfo") -> list[str]:
        """Get parameter definitions for a tool.

        Args:
            tool_info: The ToolInfo instance containing parameter information

        Returns:
            List of parameter definitions as strings in the format "name: type [= default]"
        """
        # Map JSON schema types to Python types
        type_mapping = {
            "string": "str",
            "number": "float",
            "integer": "int",
            "boolean": "bool",
            "array": "list",
            "object": "dict",
        }

        params = []
        if hasattr(tool_info, "parameters") and tool_info.parameters:
            properties = tool_info.parameters.get("properties", {})
            for param_name, param_schema in properties.items():
                # Get the type and map it to a Python type
                json_type = param_schema.get("type", "Any")
                param_type = type_mapping.get(json_type, "Any")

                # Handle array items type if specified
                if json_type == "array" and "items" in param_schema:
                    items_type = param_schema["items"].get("type", "Any")
                    param_type = f"list[{type_mapping.get(items_type, 'Any')}]"

                # Handle default value
                if "default" in param_schema:
                    default = param_schema["default"]
                    # Handle special cases for defaults
                    if default is None:
                        default_repr = "None"
                    elif isinstance(default, str):
                        default_repr = f'"{default}"'
                    else:
                        default_repr = repr(default)
                    params.append(f"{param_name}: {param_type} = {default_repr}")
                else:
                    params.append(f"{param_name}: {param_type}")

        return params

    def cleanup(self) -> None:
        """Clean up any generated temporary files.

        Note: Only cleans up directories that were created by this instance
        and not explicitly provided as output directories.
        """
        for path in list(self._generated_files):
            try:
                if path.is_dir():
                    import shutil

                    shutil.rmtree(path, ignore_errors=True)
                elif path.exists():
                    path.unlink(missing_ok=True)
                self._generated_files.remove(path)
            except Exception as e:
                logger.warning(
                    f"Failed to clean up {path}", error=str(e), exc_info=True
                )
