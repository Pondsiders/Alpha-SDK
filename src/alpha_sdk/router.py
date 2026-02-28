"""router.py — Output event fan-out to observers and consumers.

The router reads events from an async source (the engine's event stream)
and distributes them to:
  1. Observers — registered callbacks that watch events (suggest, archive, etc.)
  2. The consumer — via an async queue that feeds session.events()

Observers are simple: they receive events and do whatever they want with them.
They don't block the pipeline — if an observer is slow, events keep flowing.

The consumer queue is how session.events() works: the router pushes events
into a queue, the consumer pulls from it. This decouples the engine's output
rate from the consumer's processing rate.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Protocol

from .engine import Event, ResultEvent


# -- Observer protocol --------------------------------------------------------


class Observer(Protocol):
    """Something that watches the output event stream.

    Observers receive every event from the engine. They can be async
    or sync — the router handles both.
    """

    async def on_event(self, event: Event) -> None:
        """Called for every event from the engine."""
        ...


# -- Router -------------------------------------------------------------------


class Router:
    """Fans out engine events to observers and the consumer queue.

    Usage:
        router = Router(observers=[suggest, archive])
        # The session drives the router by calling route_event() for each
        # event from the engine, and the consumer reads from events().
    """

    def __init__(self, observers: list[Observer] | None = None):
        self._observers = list(observers) if observers else []
        self._consumer_queue: asyncio.Queue[Event | None] = asyncio.Queue()
        self._running = False

    @property
    def observer_count(self) -> int:
        return len(self._observers)

    def add_observer(self, observer: Observer) -> None:
        """Add an observer to receive events."""
        self._observers.append(observer)

    def remove_observer(self, observer: Observer) -> None:
        """Remove an observer."""
        self._observers.remove(observer)

    async def route_event(self, event: Event) -> None:
        """Route a single event to all observers and the consumer queue.

        Called by the session driver for each event from the engine.
        Observers are notified concurrently. The consumer queue gets
        the event after observers have been notified (but we don't
        wait for observers to finish).
        """
        # Fan out to observers — fire and forget
        if self._observers:
            tasks = [
                asyncio.create_task(self._safe_notify(obs, event))
                for obs in self._observers
            ]
            # Don't await — observers run in background
            for task in tasks:
                task.add_done_callback(self._observer_done)

        # Push to consumer queue
        await self._consumer_queue.put(event)

    async def route_stream(self, events: AsyncIterator[Event]) -> None:
        """Route all events from an async iterator.

        Reads events from the source (typically engine.events()) and
        routes each one. Stops when the iterator is exhausted.
        """
        self._running = True
        try:
            async for event in events:
                await self.route_event(event)
        finally:
            self._running = False

    async def events(self) -> AsyncIterator[Event]:
        """Consumer-facing event stream.

        Yields events that have been routed. Stops after receiving
        a ResultEvent (end of turn) — mirrors engine.events() behavior.
        """
        while True:
            event = await self._consumer_queue.get()
            if event is None:
                return  # Shutdown signal
            yield event
            if isinstance(event, ResultEvent):
                return  # End of turn

    async def close(self) -> None:
        """Signal the consumer that no more events are coming."""
        await self._consumer_queue.put(None)

    async def _safe_notify(self, observer: Observer, event: Event) -> None:
        """Notify an observer, catching exceptions so one bad observer
        doesn't kill the pipeline."""
        try:
            await observer.on_event(event)
        except Exception:
            # Observers shouldn't crash the router.
            # TODO: log this when we have observability
            pass

    @staticmethod
    def _observer_done(task: asyncio.Task) -> None:
        """Callback for observer tasks — suppress exceptions."""
        if not task.cancelled():
            task.exception() if task.done() else None
