"""Tests for router.py — output event fan-out."""

import asyncio

import pytest

from alpha_sdk.engine import AssistantEvent, Event, ResultEvent, SystemEvent
from alpha_sdk.router import Router


# -- Test observer implementations -------------------------------------------


class CollectingObserver:
    """Observer that collects all events for assertions."""

    def __init__(self):
        self.events: list[Event] = []

    async def on_event(self, event: Event) -> None:
        self.events.append(event)


class SlowObserver:
    """Observer that takes a while to process events."""

    def __init__(self, delay: float = 0.05):
        self.delay = delay
        self.events: list[Event] = []

    async def on_event(self, event: Event) -> None:
        await asyncio.sleep(self.delay)
        self.events.append(event)


class CrashingObserver:
    """Observer that raises exceptions."""

    async def on_event(self, event: Event) -> None:
        raise ValueError("I'm broken!")


class CountingObserver:
    """Observer that just counts events."""

    def __init__(self):
        self.count = 0

    async def on_event(self, event: Event) -> None:
        self.count += 1


# -- Router basics -----------------------------------------------------------


class TestRouterBasics:
    def test_starts_with_no_observers(self):
        router = Router()
        assert router.observer_count == 0

    def test_starts_with_observers(self):
        obs = CollectingObserver()
        router = Router(observers=[obs])
        assert router.observer_count == 1

    def test_add_observer(self):
        router = Router()
        obs = CollectingObserver()
        router.add_observer(obs)
        assert router.observer_count == 1

    def test_remove_observer(self):
        obs = CollectingObserver()
        router = Router(observers=[obs])
        router.remove_observer(obs)
        assert router.observer_count == 0


# -- Event routing -----------------------------------------------------------


class TestEventRouting:
    @pytest.mark.asyncio
    async def test_observer_receives_events(self):
        obs = CollectingObserver()
        router = Router(observers=[obs])

        event = AssistantEvent(raw={}, content=[{"type": "text", "text": "hi"}])
        await router.route_event(event)

        # Give observer task time to complete
        await asyncio.sleep(0.01)
        assert len(obs.events) == 1
        assert obs.events[0] is event

    @pytest.mark.asyncio
    async def test_multiple_observers_all_receive(self):
        obs1 = CollectingObserver()
        obs2 = CollectingObserver()
        router = Router(observers=[obs1, obs2])

        event = AssistantEvent(raw={}, content=[])
        await router.route_event(event)

        await asyncio.sleep(0.01)
        assert len(obs1.events) == 1
        assert len(obs2.events) == 1

    @pytest.mark.asyncio
    async def test_consumer_receives_events(self):
        router = Router()

        event = AssistantEvent(raw={}, content=[{"type": "text", "text": "hello"}])
        await router.route_event(event)

        result = ResultEvent(raw={}, session_id="test-123")
        await router.route_event(result)

        received = []
        async for e in router.events():
            received.append(e)

        assert len(received) == 2
        assert isinstance(received[0], AssistantEvent)
        assert isinstance(received[1], ResultEvent)

    @pytest.mark.asyncio
    async def test_consumer_stops_at_result_event(self):
        router = Router()

        await router.route_event(AssistantEvent(raw={}, content=[]))
        await router.route_event(ResultEvent(raw={}, session_id="s1"))
        # This event is AFTER the ResultEvent — consumer shouldn't see it
        await router.route_event(AssistantEvent(raw={}, content=[]))

        received = []
        async for e in router.events():
            received.append(e)

        assert len(received) == 2  # AssistantEvent + ResultEvent

    @pytest.mark.asyncio
    async def test_no_observers_still_routes_to_consumer(self):
        router = Router()  # No observers

        await router.route_event(AssistantEvent(raw={}, content=[]))
        await router.route_event(ResultEvent(raw={}))

        received = []
        async for e in router.events():
            received.append(e)

        assert len(received) == 2


# -- Observer resilience -----------------------------------------------------


class TestObserverResilience:
    @pytest.mark.asyncio
    async def test_crashing_observer_doesnt_kill_router(self):
        good = CollectingObserver()
        bad = CrashingObserver()
        router = Router(observers=[bad, good])

        event = AssistantEvent(raw={}, content=[])
        await router.route_event(event)
        await router.route_event(ResultEvent(raw={}))

        await asyncio.sleep(0.01)
        # Good observer still got the event despite bad one crashing
        assert len(good.events) == 2

        # Consumer still gets events
        received = []
        async for e in router.events():
            received.append(e)
        assert len(received) == 2

    @pytest.mark.asyncio
    async def test_slow_observer_doesnt_block_consumer(self):
        slow = SlowObserver(delay=0.1)
        router = Router(observers=[slow])

        await router.route_event(AssistantEvent(raw={}, content=[]))
        await router.route_event(ResultEvent(raw={}))

        # Consumer should get events immediately, not after 0.1s
        received = []
        async for e in router.events():
            received.append(e)

        assert len(received) == 2
        # Slow observer may not have finished yet — that's fine
        # It'll catch up in its own time


# -- route_stream -------------------------------------------------------------


class TestRouteStream:
    @pytest.mark.asyncio
    async def test_route_stream(self):
        obs = CollectingObserver()
        router = Router(observers=[obs])

        async def fake_events():
            yield AssistantEvent(raw={}, content=[{"type": "text", "text": "hi"}])
            yield SystemEvent(raw={}, subtype="session_id")
            yield ResultEvent(raw={}, session_id="s1")

        # Route in background, consume in foreground
        route_task = asyncio.create_task(router.route_stream(fake_events()))

        received = []
        async for e in router.events():
            received.append(e)

        await route_task

        assert len(received) == 3
        await asyncio.sleep(0.01)
        assert len(obs.events) == 3


# -- Close/shutdown -----------------------------------------------------------


class TestRouterClose:
    @pytest.mark.asyncio
    async def test_close_signals_consumer(self):
        router = Router()
        await router.close()

        received = []
        async for e in router.events():
            received.append(e)

        assert len(received) == 0  # Consumer exits immediately


# -- is_replay passthrough ---------------------------------------------------


class TestReplayPassthrough:
    @pytest.mark.asyncio
    async def test_replay_events_reach_observer(self):
        obs = CollectingObserver()
        router = Router(observers=[obs])

        event = AssistantEvent(
            raw={},
            content=[{"type": "text", "text": "replayed"}],
            is_replay=True,
        )
        await router.route_event(event)
        await router.route_event(ResultEvent(raw={}, is_replay=True))

        await asyncio.sleep(0.01)
        assert obs.events[0].is_replay is True

    @pytest.mark.asyncio
    async def test_replay_events_reach_consumer(self):
        router = Router()

        event = AssistantEvent(raw={}, content=[], is_replay=True)
        await router.route_event(event)
        await router.route_event(ResultEvent(raw={}, is_replay=True))

        received = []
        async for e in router.events():
            received.append(e)

        assert received[0].is_replay is True
        assert received[1].is_replay is True
