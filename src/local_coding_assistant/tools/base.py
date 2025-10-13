from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel


class Tool(ABC):
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

    @abstractmethod
    def run(self, payload: Input) -> Output:
        raise NotImplementedError
