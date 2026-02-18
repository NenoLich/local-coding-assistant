"""Configuration dependencies for the hybrid configuration system.

This module provides dependency-related classes that are used across
the configuration system without creating circular imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    pass


class SettingDependency(BaseModel):
    """Metadata about setting dependencies."""

    # Use regular Field to avoid circular dependency
    value_requirements: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Requirements per setting value (unified modules and capabilities)",
    )
    fallback_order: list[str] = Field(
        default_factory=list, description="Fallback values in order of preference"
    )

    def __init_subclass__(cls, section: str | None = None, **kwargs):
        """Allow section to be specified in class definition."""
        super().__init_subclass__(**kwargs)
        if section is not None:
            cls._section = section

    @classmethod
    def get_section(cls) -> str | None:
        """Get the section for this model."""
        return getattr(cls, "_section", None)
