"""Tests for event pipeline fan-out functionality."""

import asyncio
from unittest.mock import MagicMock

import pytest

from local_coding_assistant.runtime.events.contracts import (
    FrameStartEvent,
    LLMCompleteEvent,
    TurnStartEvent,
)
from local_coding_assistant.runtime.events.pipeline import (
    EventPipeline,
    EventSubscriber,
)


class MockSubscriber(EventSubscriber):
    """Mock subscriber for testing."""

    def __init__(self):
        self.events = []
        self.closed = False

    async def on_event(self, event):
        """Store received events."""
        self.events.append(event)

    async def close(self):
        """Mark as closed."""
        self.closed = True


class FailingSubscriber(EventSubscriber):
    """Subscriber that always fails."""

    def __init__(self):
        self.call_count = 0

    async def on_event(self, event):
        """Always raise an error."""
        self.call_count += 1
        raise RuntimeError("Subscriber failed")

    async def close(self):
        """No-op close."""
        pass


class TestEventPipeline:
    """Test EventPipeline fan-out functionality."""

    @pytest.fixture
    def pipeline(self):
        """Create a fresh pipeline for each test."""
        return EventPipeline()

    @pytest.fixture
    def sample_event(self):
        """Create a sample event."""
        return TurnStartEvent(
            trace_id="tr_123",
            turn_id="turn_456",
            session_id="sess_789",
            user_input="test query",
            turn_index=0,
        )

    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initializes with empty subscriber list."""
        assert pipeline.subscribers == []
        assert len(pipeline.subscribers) == 0

    def test_subscribe(self, pipeline):
        """Test subscribing a subscriber."""
        subscriber = MockSubscriber()
        pipeline.subscribe(subscriber)

        assert len(pipeline.subscribers) == 1
        assert pipeline.subscribers[0] == subscriber

    def test_subscribe_multiple(self, pipeline):
        """Test subscribing multiple subscribers."""
        sub1 = MockSubscriber()
        sub2 = MockSubscriber()
        sub3 = MockSubscriber()

        pipeline.subscribe(sub1)
        pipeline.subscribe(sub2)
        pipeline.subscribe(sub3)

        assert len(pipeline.subscribers) == 3
        assert pipeline.subscribers == [sub1, sub2, sub3]

    def test_unsubscribe(self, pipeline):
        """Test unsubscribing a subscriber."""
        subscriber = MockSubscriber()
        pipeline.subscribe(subscriber)
        assert len(pipeline.subscribers) == 1

        pipeline.unsubscribe(subscriber)
        assert len(pipeline.subscribers) == 0

    def test_unsubscribe_nonexistent(self, pipeline):
        """Test unsubscribing a subscriber that isn't registered."""
        subscriber = MockSubscriber()
        # Should not raise
        pipeline.unsubscribe(subscriber)
        assert len(pipeline.subscribers) == 0

    @pytest.mark.asyncio
    async def test_emit_to_single_subscriber(self, pipeline, sample_event):
        """Test emitting event to single subscriber."""
        subscriber = MockSubscriber()
        pipeline.subscribe(subscriber)

        await pipeline.emit(sample_event)

        assert len(subscriber.events) == 1
        assert subscriber.events[0] == sample_event

    @pytest.mark.asyncio
    async def test_emit_to_multiple_subscribers(self, pipeline, sample_event):
        """Test emitting event to multiple subscribers concurrently."""
        sub1 = MockSubscriber()
        sub2 = MockSubscriber()
        sub3 = MockSubscriber()

        pipeline.subscribe(sub1)
        pipeline.subscribe(sub2)
        pipeline.subscribe(sub3)

        await pipeline.emit(sample_event)

        # All subscribers should receive the event
        assert len(sub1.events) == 1
        assert len(sub2.events) == 1
        assert len(sub3.events) == 1
        assert sub1.events[0] == sample_event
        assert sub2.events[0] == sample_event
        assert sub3.events[0] == sample_event

    @pytest.mark.asyncio
    async def test_emit_multiple_events(self, pipeline):
        """Test emitting multiple events in sequence."""
        subscriber = MockSubscriber()
        pipeline.subscribe(subscriber)

        event1 = TurnStartEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            session_id="sess_1",
            user_input="query 1",
            turn_index=0,
        )
        event2 = FrameStartEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            frame_id="frame_1",
            session_id="sess_1",
            iteration=1,
            context_snapshot_id="ctx_1",
            context_hash="sha256:abc",
        )
        event3 = LLMCompleteEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            frame_id="frame_1",
            session_id="sess_1",
            content="response",
            total_tokens=100,
        )

        await pipeline.emit(event1)
        await pipeline.emit(event2)
        await pipeline.emit(event3)

        assert len(subscriber.events) == 3
        assert subscriber.events[0] == event1
        assert subscriber.events[1] == event2
        assert subscriber.events[2] == event3

    @pytest.mark.asyncio
    async def test_emit_with_no_subscribers(self, pipeline, sample_event):
        """Test emitting event when no subscribers are registered."""
        # Should not raise
        await pipeline.emit(sample_event)

    @pytest.mark.asyncio
    async def test_emit_with_failing_subscriber(self, pipeline, sample_event):
        """Test that one failing subscriber doesn't affect others."""
        failing_sub = FailingSubscriber()
        good_sub = MockSubscriber()

        pipeline.subscribe(failing_sub)
        pipeline.subscribe(good_sub)

        # Should not raise despite failing subscriber
        await pipeline.emit(sample_event)

        # Failing subscriber was called
        assert failing_sub.call_count == 1
        # Good subscriber still received the event
        assert len(good_sub.events) == 1
        assert good_sub.events[0] == sample_event

    @pytest.mark.asyncio
    async def test_emit_concurrent_to_subscribers(self, pipeline, sample_event):
        """Test that events are emitted to subscribers concurrently."""
        # Track call order
        call_order = []

        class OrderedSubscriber(MockSubscriber):
            def __init__(self, name):
                super().__init__()
                self.name = name

            async def on_event(self, event):
                call_order.append(self.name)
                # Small delay to test concurrency
                await asyncio.sleep(0.01)
                await super().on_event(event)

        sub1 = OrderedSubscriber("sub1")
        sub2 = OrderedSubscriber("sub2")
        sub3 = OrderedSubscriber("sub3")

        pipeline.subscribe(sub1)
        pipeline.subscribe(sub2)
        pipeline.subscribe(sub3)

        await pipeline.emit(sample_event)

        # All should have been called
        assert len(call_order) == 3
        # All should have received the event
        assert len(sub1.events) == 1
        assert len(sub2.events) == 1
        assert len(sub3.events) == 1

    @pytest.mark.asyncio
    async def test_emit_assigns_seq_before_fan_out(self, pipeline):
        """Pipeline assigns seq before fan-out so all subscribers observe same seq."""
        sub1 = MockSubscriber()
        sub2 = MockSubscriber()
        pipeline.subscribe(sub1)
        pipeline.subscribe(sub2)

        event = TurnStartEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            session_id="sess_1",
            user_input="query",
            turn_index=0,
        )

        await pipeline.emit(event)

        assert sub1.events[0].seq == 1
        assert sub2.events[0].seq == 1
        assert event.seq == 1

    @pytest.mark.asyncio
    async def test_emit_seq_monotonic_per_session(self, pipeline):
        """Seq increments monotonically per session across multiple emits."""
        subscriber = MockSubscriber()
        pipeline.subscribe(subscriber)

        session_id = "sess_1"
        events = [
            TurnStartEvent(
                trace_id=f"tr_{i}",
                turn_id=f"turn_{i}",
                session_id=session_id,
                user_input=f"query {i}",
                turn_index=0,
            )
            for i in range(3)
        ]

        for event in events:
            await pipeline.emit(event)

        seqs = [subscriber.events[i].seq for i in range(3)]
        assert seqs == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_emit_different_sessions_independent_counters(self, pipeline):
        """Different sessions maintain independent seq counters."""
        subscriber = MockSubscriber()
        pipeline.subscribe(subscriber)

        event_a = TurnStartEvent(
            trace_id="tr_a",
            turn_id="turn_a",
            session_id="sess_a",
            user_input="query a",
            turn_index=0,
        )
        event_b = TurnStartEvent(
            trace_id="tr_b",
            turn_id="turn_b",
            session_id="sess_b",
            user_input="query b",
            turn_index=0,
        )
        event_c = TurnStartEvent(
            trace_id="tr_c",
            turn_id="turn_c",
            session_id="sess_a",
            user_input="query c",
            turn_index=0,
        )

        await pipeline.emit(event_a)
        await pipeline.emit(event_b)
        await pipeline.emit(event_c)

        assert subscriber.events[0].seq == 1  # sess_a first
        assert subscriber.events[1].seq == 1  # sess_b first
        assert subscriber.events[2].seq == 2  # sess_a second

    @pytest.mark.asyncio
    async def test_emit_with_execution_store_seeds_counter(self, pipeline):
        """Pipeline seeds seq counter from store's get_max_seq on first use."""
        mock_store = MagicMock()

        async def fake_get_max_seq(session_id: str) -> int:
            return 10

        mock_store.get_max_seq = fake_get_max_seq
        pipeline._execution_store = mock_store

        subscriber = MockSubscriber()
        pipeline.subscribe(subscriber)

        event = TurnStartEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            session_id="sess_1",
            user_input="query",
            turn_index=0,
        )

        await pipeline.emit(event)

        assert subscriber.events[0].seq == 11  # store max 10 + 1

    @pytest.mark.asyncio
    async def test_emit_no_session_id_keeps_seq_zero(self, pipeline):
        """Events without session_id keep seq=0."""
        subscriber = MockSubscriber()
        pipeline.subscribe(subscriber)

        event = TurnStartEvent(
            trace_id="tr_1",
            turn_id="turn_1",
            session_id="",
            user_input="query",
            turn_index=0,
        )

        await pipeline.emit(event)

        assert subscriber.events[0].seq == 0
        assert event.seq == 0

    @pytest.mark.asyncio
    async def test_close(self, pipeline):
        """Test closing pipeline closes all subscribers."""
        sub1 = MockSubscriber()
        sub2 = MockSubscriber()

        pipeline.subscribe(sub1)
        pipeline.subscribe(sub2)

        await pipeline.close()

        assert sub1.closed is True
        assert sub2.closed is True
        assert len(pipeline.subscribers) == 0

    @pytest.mark.asyncio
    async def test_close_with_failing_subscriber(self, pipeline):
        """Test that close handles failing subscribers gracefully."""

        class FailingCloseSubscriber(MockSubscriber):
            async def close(self):
                raise RuntimeError("Close failed")

        sub1 = MockSubscriber()
        sub2 = FailingCloseSubscriber()
        sub3 = MockSubscriber()

        pipeline.subscribe(sub1)
        pipeline.subscribe(sub2)
        pipeline.subscribe(sub3)

        # Should not raise
        await pipeline.close()

        # Subscribers that could close should be closed
        assert sub1.closed is True
        assert sub3.closed is True
        # List should be cleared regardless
        assert len(pipeline.subscribers) == 0

    @pytest.mark.asyncio
    async def test_subscribers_property_returns_copy(self, pipeline):
        """Test that subscribers property returns a copy."""
        subscriber = MockSubscriber()
        pipeline.subscribe(subscriber)

        subs = pipeline.subscribers
        subs.append(MockSubscriber())

        # Original list should not be modified
        assert len(pipeline.subscribers) == 1


class TestCreateDefaultPipeline:
    """Test create_default_pipeline factory function."""

    @pytest.mark.asyncio
    async def test_create_default_pipeline_no_args(self):
        """Test creating default pipeline with no arguments."""
        from local_coding_assistant.runtime.events.pipeline import (
            create_default_pipeline,
        )

        pipeline = create_default_pipeline()

        assert isinstance(pipeline, EventPipeline)
        # With no execution_store, should have no store subscriber
        # With enable_otel=False, should have no otel subscriber
        assert len(pipeline.subscribers) == 0

        await pipeline.close()

    @pytest.mark.asyncio
    async def test_create_default_pipeline_with_store(self):
        """Test creating default pipeline with execution store."""
        from local_coding_assistant.runtime.events.pipeline import (
            create_default_pipeline,
        )

        mock_store = MagicMock()

        pipeline = create_default_pipeline(execution_store=mock_store)

        assert isinstance(pipeline, EventPipeline)
        # Should have store subscriber
        assert len(pipeline.subscribers) == 1

        await pipeline.close()

    @pytest.mark.asyncio
    async def test_create_default_pipeline_with_otel_unavailable(self):
        """Test creating default pipeline with OTel when unavailable."""
        from local_coding_assistant.runtime.events.pipeline import (
            create_default_pipeline,
        )

        # OTel subscriber import should fail gracefully
        pipeline = create_default_pipeline(enable_otel=True)

        assert isinstance(pipeline, EventPipeline)
        # Should not have OTel subscriber if import fails
        # (This test assumes OTel is not installed in test env)

        await pipeline.close()
