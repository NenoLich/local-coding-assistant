from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from local_coding_assistant.tools.tool_registry import Tool


class SumInput(BaseModel):
    a: int = Field(..., ge=-10_000, le=10_000)
    b: int = Field(..., ge=-10_000, le=10_000)


class SumOutput(BaseModel):
    sum: int


class SumTool(Tool):
    """Compute a + b and return the sum."""

    InputModel = SumInput
    OutputModel = SumOutput

    def __init__(self) -> None:
        super().__init__(name="sum")

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        a = int(payload["a"])  # validated by InputModel
        b = int(payload["b"])  # validated by InputModel
        return {"sum": a + b}
