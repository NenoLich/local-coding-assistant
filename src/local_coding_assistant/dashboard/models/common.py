"""Common Pydantic models for the dashboard."""

from typing import Any, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class PaginationParams(BaseModel):
    """Pagination parameters."""

    limit: int = Field(default=50, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


class PaginatedResponse(BaseModel):
    """Generic paginated response."""

    items: list[Any]
    total: int
    offset: int
    limit: int
    has_next: bool
    has_prev: bool


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    detail: str | None = None


class StatusSummary(BaseModel):
    """Aggregate status summary."""

    total: int
    running: int
    completed: int
    error: int
    idle: int
