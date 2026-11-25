"""Execution backend implementations."""

from local_coding_assistant.tools_platform.backends.base import ExecutionBackend
from local_coding_assistant.tools_platform.backends.match_paths_impl import (
    match_paths,
)
from local_coding_assistant.tools_platform.backends.search_text_impl import (
    search_with_python,
    search_with_ripgrep,
)

__all__ = [
    "ExecutionBackend",
    "match_paths",
    "search_with_python",
    "search_with_ripgrep",
]
