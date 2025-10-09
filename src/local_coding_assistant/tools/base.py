from __future__ import annotations

from pydantic import BaseModel


class Tool:
    """Base contract for tools with Pydantic I/O schemas.

    Subclasses should override `name`, `description`, nested `Input`/`Output`
    models, and the `run()` implementation.
    """

    name: str
    description: str

    class Input(BaseModel):
        pass

    class Output(BaseModel):
        pass

    def run(self, payload: Input) -> Output:  # type: ignore[type-arg]
        raise NotImplementedError
