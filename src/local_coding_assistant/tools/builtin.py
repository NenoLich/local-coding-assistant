from __future__ import annotations

from pydantic import BaseModel, Field

from local_coding_assistant.tools.base import Tool


class SumTool(Tool):
    name = "sum"
    description = "Compute a + b and return the sum."

    class Input(BaseModel):
        a: int = Field(..., ge=-10_000, le=10_000)
        b: int = Field(..., ge=-10_000, le=10_000)

    class Output(BaseModel):
        sum: int

    def run(self, payload: SumTool.Input) -> SumTool.Output:
        return self.Output(sum=payload.a + payload.b)
