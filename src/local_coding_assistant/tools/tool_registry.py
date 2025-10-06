"""Tool registry and tool interfaces."""

from collections.abc import Iterable, Iterator
from typing import Any


class ToolRegistry(Iterable[Any]):
    """Simple in-memory registry of tools.

    Later, tools can be schema-defined classes with input/output models.
    """

    def __init__(self) -> None:
        self._tools: list[Any] = []

    def register(self, tool: Any) -> None:
        self._tools.append(tool)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._tools)

    def __len__(self) -> int:  # so `len(tools)` works
        return len(self._tools)
