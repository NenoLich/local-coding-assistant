"""
Test fixtures and utilities for streaming tests.
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from local_coding_assistant.providers.base import BaseDriver, ProviderLLMRequest, ProviderLLMResponse
from local_coding_assistant.runtime.events import EventType, ExecutionEvent


class MockStreamingDriver(BaseDriver):
    """Mock driver for testing streaming functionality."""

    def __init__(self, events_to_emit: list[ExecutionEvent] | None = None, **kwargs):
        super().__init__(api_key=None, base_url="mock://", **kwargs)
        self.events_to_emit = events_to_emit or []
        self.generate_called = False
        self.stream_called = False

    async def generate(self, request: ProviderLLMRequest) -> ProviderLLMResponse:
        """Mock generate method."""
        self.generate_called = True
        return ProviderLLMResponse(
            content="Mock response",
            model=request.model,
            tokens_used=10,
        )

    async def stream(
        self, request: ProviderLLMRequest
    ) -> AsyncIterator[ExecutionEvent]:
        """Mock stream method that yields predefined events."""
        self.stream_called = True
        for event in self.events_to_emit:
            yield event
            await asyncio.sleep(0.001)  # Small delay to simulate real streaming

    async def health_check(self) -> bool:
        """Mock health check."""
        return True


async def collect_events(event_iterator: AsyncIterator[ExecutionEvent]) -> list[ExecutionEvent]:
    """Collect all events from an async iterator into a list."""
    events = []
    async for event in event_iterator:
        events.append(event)
    return events


async def collect_events_with_timeout(
    event_iterator: AsyncIterator[ExecutionEvent],
    timeout: float = 5.0
) -> list[ExecutionEvent]:
    """Collect events with a timeout to prevent hanging tests."""
    events = []

    async def collect():
        async for event in event_iterator:
            events.append(event)

    try:
        await asyncio.wait_for(collect(), timeout=timeout)
    except asyncio.TimeoutError:
        pass  # Timeout is expected for infinite iterators in tests

    return events


@pytest.fixture
def mock_streaming_driver():
    """Fixture providing a basic mock streaming driver."""
    return MockStreamingDriver()


@pytest.fixture
def mock_streaming_driver_with_events(sample_events):
    """Fixture providing a mock streaming driver with predefined events."""
    return MockStreamingDriver(events_to_emit=sample_events)


@pytest.fixture
def sample_events():
    """Fixture providing sample events for testing."""
    return [
        ExecutionEvent(
            type=EventType.TURN_START,
            session_id="test_session",
            data={"user_input": "test query"},
        ),
        ExecutionEvent(
            type=EventType.FRAME_START,
            session_id="test_session",
            frame_id="frame_1",
            data={"iteration": 1},
        ),
        ExecutionEvent(
            type=EventType.LLM_START,
            session_id="test_session",
            frame_id="frame_1",
            data={"model": "test-model"},
        ),
        ExecutionEvent(
            type=EventType.LLM_CHUNK,
            session_id="test_session",
            frame_id="frame_1",
            data={"content": "Hello"},
        ),
        ExecutionEvent(
            type=EventType.LLM_CHUNK,
            session_id="test_session",
            frame_id="frame_1",
            data={"content": " world"},
        ),
        ExecutionEvent(
            type=EventType.LLM_COMPLETE,
            session_id="test_session",
            frame_id="frame_1",
            data={"finish_reason": "stop"},
        ),
        ExecutionEvent(
            type=EventType.FRAME_COMPLETE,
            session_id="test_session",
            frame_id="frame_1",
            data={"status": "success"},
        ),
        ExecutionEvent(
            type=EventType.TURN_COMPLETE,
            session_id="test_session",
            data={"report": "test report"},
        ),
    ]


@pytest.fixture
def mock_event_stream(sample_events):
    """Fixture providing a mock async iterator of events."""
    async def event_generator():
        for event in sample_events:
            yield event
            await asyncio.sleep(0.01)  # Small delay to simulate real streaming

    return event_generator()


@pytest.fixture
def mock_provider_llm_request():
    """Fixture providing a mock ProviderLLMRequest for testing."""
    return ProviderLLMRequest(
        messages=[
            {"role": "user", "content": "Test message"},
        ],
        model="test-model",
    )
