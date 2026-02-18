"""
Base handler for partial responses.

This module provides the abstract base class and common functionality
for handling partial responses due to truncation or tool failures.
"""

from abc import ABC, abstractmethod

from local_coding_assistant.runtime.handlers.handler_types import (
    HandlerContext,
    HandlerOutput,
)


class PartialResponseHandler(ABC):
    """Base class for handling partial responses."""

    @abstractmethod
    async def handle(self, context: HandlerContext) -> HandlerOutput:
        """Handle a partial response based on the provided context."""
        pass
