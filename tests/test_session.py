"""Tests for session.py — duplex channel."""

import asyncio

import pytest

from alpha_sdk.engine import (
    AssistantEvent,
    Engine,
    EngineState,
    Event,
    ResultEvent,
)
from alpha_sdk.queue import MessageQueue
from alpha_sdk.router import Router
from alpha_sdk.session import Session


# -- Mock engine for unit testing -------------------------------------------


class MockEngine:
    """A mock engine that simulates claude's behavior without a subprocess."""

    def __init__(self, responses: list[list[Event]] | None = None):
        """
        Args:
            responses: List of event lists. Each list is the response to
                       one send() call. Consumed in order.
        """
        self._responses = list(responses) if responses else []
        self._state = EngineState.IDLE
        self._session_id = None
        self._sent: list[str] = []

    @property
    def state(self) -> EngineState:
        return self._state

    @property
    def session_id(self):
        return self._session_id

    async def start(self, session_id=None):
        self._state = EngineState.READY
        self._session_id = session_id or "mock-session-id"

    async def send(self, text: str):
        self._sent.append(text)

    async def events(self):
        if self._responses:
            response = self._responses.pop(0)
            for event in response:
                yield event
        else:
            # Default: return a simple assistant + result
            yield AssistantEvent(
                raw={}, content=[{"type": "text", "text": "mock response"}]
            )
            yield ResultEvent(raw={}, session_id=self._session_id)

    async def stop(self):
        self._state = EngineState.STOPPED


# -- Session basics ----------------------------------------------------------


class TestSessionBasics:
    @pytest.mark.asyncio
    async def test_start_is_clean(self):
        engine = MockEngine()
        session = Session(engine=engine)
        await session.start()
        # Session ID comes from engine (set during mock start)
        assert session.session_id == "mock-session-id"
        await session.stop()

    @pytest.mark.asyncio
    async def test_session_id_property(self):
        engine = MockEngine()
        session = Session(engine=engine)
        await session.start()
        assert session.session_id == "mock-session-id"
        await session.stop()

    @pytest.mark.asyncio
    async def test_send_and_receive(self):
        engine = MockEngine()
        session = Session(engine=engine)
        await session.start()

        await session.send("Hello!")

        received = []
        async for event in session.events():
            received.append(event)

        assert len(received) == 2
        assert isinstance(received[0], AssistantEvent)
        assert received[0].text == "mock response"
        assert isinstance(received[1], ResultEvent)
        await session.stop()

    @pytest.mark.asyncio
    async def test_multiple_turns(self):
        responses = [
            [
                AssistantEvent(raw={}, content=[{"type": "text", "text": "first"}]),
                ResultEvent(raw={}, session_id="s1"),
            ],
            [
                AssistantEvent(raw={}, content=[{"type": "text", "text": "second"}]),
                ResultEvent(raw={}, session_id="s1"),
            ],
        ]
        engine = MockEngine(responses=responses)
        session = Session(engine=engine)
        await session.start()

        # Turn 1
        await session.send("msg 1")
        events1 = []
        async for e in session.events():
            events1.append(e)
        assert events1[0].text == "first"

        # Turn 2
        await session.send("msg 2")
        events2 = []
        async for e in session.events():
            events2.append(e)
        assert events2[0].text == "second"

        await session.stop()


# -- Session with observers --------------------------------------------------


class CollectingObserver:
    def __init__(self):
        self.events = []

    async def on_event(self, event):
        self.events.append(event)


class TestSessionObservers:
    @pytest.mark.asyncio
    async def test_observer_receives_events(self):
        obs = CollectingObserver()
        engine = MockEngine()
        router = Router(observers=[obs])
        session = Session(engine=engine, router=router)
        await session.start()

        await session.send("Hello!")
        async for _ in session.events():
            pass

        await asyncio.sleep(0.05)  # Let observer tasks complete
        assert len(obs.events) == 2
        await session.stop()


# -- Session with custom queue -----------------------------------------------


class TestSessionQueue:
    @pytest.mark.asyncio
    async def test_custom_queue(self):
        queue = MessageQueue(maxsize=5)
        engine = MockEngine()
        session = Session(engine=engine, queue=queue)
        await session.start()

        await session.send("via custom queue")
        async for _ in session.events():
            pass

        assert "via custom queue" in engine._sent
        await session.stop()


# -- Session stop/cleanup ---------------------------------------------------


class TestSessionStop:
    @pytest.mark.asyncio
    async def test_stop_cleans_up(self):
        engine = MockEngine()
        session = Session(engine=engine)
        await session.start()

        await session.stop()
        assert engine.state == EngineState.STOPPED

    @pytest.mark.asyncio
    async def test_stop_without_activity(self):
        """Stopping a session that never sent anything should be clean."""
        engine = MockEngine()
        session = Session(engine=engine)
        await session.start()
        await session.stop()
        assert engine.state == EngineState.STOPPED


# -- Session replay ----------------------------------------------------------


class TestSessionReplay:
    @pytest.mark.asyncio
    async def test_replay_routes_events(self, tmp_path):
        """Resuming a session replays history through the router."""
        import json

        session_id = "replay-test-123"
        jsonl_path = tmp_path / f"{session_id}.jsonl"
        records = [
            {
                "type": "assistant",
                "sessionId": session_id,
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Previous message"}],
                },
            },
        ]
        with open(jsonl_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        obs = CollectingObserver()
        engine = MockEngine()
        router = Router(observers=[obs])
        session = Session(engine=engine, router=router)

        # Monkey-patch find_session_path for this test
        import alpha_sdk.session as session_module
        original_find = session_module.find_session_path
        session_module.find_session_path = lambda sid, **kw: jsonl_path if sid == session_id else None

        try:
            await session.start(session_id=session_id)

            # Consume replay events from the router
            replay_events = []
            # The replay events should be in the consumer queue
            # Read AssistantEvent + ResultEvent from replay
            async for event in session.events():
                replay_events.append(event)

            assert len(replay_events) == 2
            assert isinstance(replay_events[0], AssistantEvent)
            assert replay_events[0].is_replay is True
            assert replay_events[0].text == "Previous message"
            assert isinstance(replay_events[1], ResultEvent)
            assert replay_events[1].is_replay is True
        finally:
            session_module.find_session_path = original_find
            await session.stop()

    @pytest.mark.asyncio
    async def test_replay_missing_file_is_ok(self):
        """Resuming with no JSONL file is a no-op (new session)."""
        engine = MockEngine()
        session = Session(engine=engine)

        import alpha_sdk.session as session_module
        original_find = session_module.find_session_path
        session_module.find_session_path = lambda sid, **kw: None

        try:
            await session.start(session_id="nonexistent")
            # Should start fine, just no replay events
            assert session.session_id == "nonexistent"
        finally:
            session_module.find_session_path = original_find
            await session.stop()


# -- Producer name passthrough -----------------------------------------------


class TestProducerName:
    @pytest.mark.asyncio
    async def test_producer_name_in_send(self):
        engine = MockEngine()
        session = Session(engine=engine)
        await session.start()

        await session.send("hello", producer="test-producer")
        async for _ in session.events():
            pass

        # The message was sent to the engine
        assert "hello" in engine._sent
        await session.stop()
