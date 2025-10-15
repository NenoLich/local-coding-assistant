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


class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Provide a final answer and stop the agent loop."

    class Input(BaseModel):
        answer: str = Field(..., description="The final answer to provide")
        reasoning: str | None = Field(
            default=None, description="Optional reasoning for the answer"
        )

    class Output(BaseModel):
        final_answer: str
        reasoning: str | None = None
        stopped: bool

    def run(self, payload: FinalAnswerTool.Input) -> FinalAnswerTool.Output:
        return self.Output(
            final_answer=payload.answer, reasoning=payload.reasoning, stopped=True
        )
