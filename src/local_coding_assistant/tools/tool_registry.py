"""Tool registry and tool interfaces."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from pydantic import ValidationError

from local_coding_assistant.tools.base import Tool


class ToolRegistry(Iterable[Any]):
    """In-memory registry of tools with optional schema validation.

    You can register any object. If it subclasses `Tool`, `invoke()` will validate
    inputs/outputs against its Pydantic models when present.
    """

    def __init__(self) -> None:
        self._tools: list[Any] = []
        self._by_name: dict[str, Any] = {}

    def register(self, tool: Any) -> None:
        self._tools.append(tool)
        name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
        if isinstance(name, str):
            self._by_name[name] = tool

    def get(self, name: str) -> Any | None:
        return self._by_name.get(name)

    def invoke(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        tool = self.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool: {name}")

        # If tool adheres to Tool contract, validate
        if isinstance(tool, Tool):
            data = payload
            if tool.Input is not None:
                try:
                    data_model = tool.Input(**payload)
                    data = data_model.model_dump()
                except ValidationError as e:
                    raise ValueError(f"Invalid input for {name}: {e}") from e
            result = tool.run(data)
            if tool.OutputModel is not None:
                try:
                    out_model = tool.Output(**result)
                    return out_model.model_dump()
                except ValidationError as e:
                    raise ValueError(f"Invalid output from {name}: {e}") from e
            return result

        # Fallback: call callable or return static
        if callable(tool):
            return tool(payload)  # type: ignore[call-arg]
        return {"result": tool}

    def list(self) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        for t in self._tools:
            name = getattr(t, "name", None) or getattr(
                t, "__name__", t.__class__.__name__
            )
            items.append({"name": str(name), "type": t.__class__.__name__})
        return items

    def __iter__(self) -> Iterator[Any]:
        return iter(self._tools)

    def __len__(self) -> int:  # so `len(tools)` works
        return len(self._tools)
