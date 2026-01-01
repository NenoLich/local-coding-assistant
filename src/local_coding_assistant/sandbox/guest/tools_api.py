# type: ignore
from typing import Any, Dict, Optional, Union, List, AsyncIterator
import asyncio
from resource_tracker import tracker

class ToolError(Exception):
    """Base exception for tool execution errors."""
    pass

# Tool imports
import sys
import os
# Add /tools to Python path for imports
if '/tools' not in sys.path:
    sys.path.insert(0, '/tools')

from sandbox_tools import MathTool
from sandbox_tools import ListFilesInCwd
from sandbox_tools import FinalAnswerTool


async def math_async(operation: str, numbers: list[float]) -> Any:
    """Async version of math function with resource tracking."""
    tool = MathTool()
    run_kwargs = {}
    if 'operation' in locals() and operation is not None:
        run_kwargs['operation'] = operation
    if 'numbers' in locals() and numbers is not None:
        run_kwargs['numbers'] = numbers
    @tracker.track(tool_name="math")
    async def _execute():
        try:
            return await tool.run(**run_kwargs)
        except Exception as e:
            raise ToolError(f"Error in math: {str(e)}") from e

    return await _execute()

def math(operation: str, numbers: list[float]) -> Any:
    """Synchronous version of math function."""
    loop = None
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(math_async(operation, numbers))
    except Exception as e:
        raise ToolError(f"Error in sync wrapper for math: {str(e)}") from e
    finally:
        loop.close()

async def file_list_async() -> Any:
    """Async version of file_list function with resource tracking."""
    tool = ListFilesInCwd()
    run_kwargs = {}
    @tracker.track(tool_name="file_list")
    async def _execute():
        try:
            return await tool.run(**run_kwargs)
        except Exception as e:
            raise ToolError(f"Error in file_list: {str(e)}") from e

    return await _execute()

def file_list() -> Any:
    """Synchronous version of file_list function."""
    loop = None
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(file_list_async())
    except Exception as e:
        raise ToolError(f"Error in sync wrapper for file_list: {str(e)}") from e
    finally:
        loop.close()

async def final_answer_async(answer: str, answer_format: str = "text", metadata: dict = None) -> Any:
    """Async version of final_answer function with resource tracking."""
    tool = FinalAnswerTool()
    run_kwargs = {}
    if 'answer' in locals() and answer is not None:
        run_kwargs['answer'] = answer
    if 'answer_format' in locals() and answer_format is not None:
        run_kwargs['answer_format'] = answer_format
    if 'metadata' in locals() and metadata is not None:
        run_kwargs['metadata'] = metadata
    @tracker.track(tool_name="final_answer")
    async def _execute():
        try:
            return await tool.run(**run_kwargs)
        except Exception as e:
            raise ToolError(f"Error in final_answer: {str(e)}") from e

    return await _execute()

def final_answer(answer: str, answer_format: str = "text", metadata: dict = None) -> Any:
    """Synchronous version of final_answer function."""
    loop = None
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(final_answer_async(answer, answer_format, metadata))
    except Exception as e:
        raise ToolError(f"Error in sync wrapper for final_answer: {str(e)}") from e
    finally:
        loop.close()
