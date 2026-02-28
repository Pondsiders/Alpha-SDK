"""session.py — Duplex channel. send() and events().

The session is the main API surface for consumers. It composes:
  - Engine: the claude subprocess
  - Queue: input messages from producers
  - Router: output event fan-out to observers and consumers

Two independent streams:
  - Producers call send() to queue messages
  - Consumers iterate events() to read responses

The session driver runs in the background, pulling from the queue,
sending to the engine, and routing events through the router.

Usage:
    session = Session(engine=engine, router=router)
    session_id = await session.start()

    # Producer side
    await session.send("Hello!")

    # Consumer side
    async for event in session.events():
        ...

    await session.stop()
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator

from .engine import Engine, EngineState, Event, ResultEvent
from .queue import Message, MessageQueue
from .replay import replay_session, find_session_path
from .router import Observer, Router


class Session:
    """Duplex channel wrapping engine + queue + router.

    The session is the primary API for consumers. It handles:
    - Starting/stopping the engine
    - Driving messages from the queue to the engine
    - Routing engine events to observers and consumers
    - Session replay for resume
    """

    def __init__(
        self,
        engine: Engine,
        router: Router | None = None,
        queue: MessageQueue | None = None,
    ):
        self._engine = engine
        self._router = router or Router()
        self._queue = queue or MessageQueue()
        self._driver_task: asyncio.Task | None = None
        self._session_id: str | None = None

    @property
    def engine(self) -> Engine:
        """The underlying engine."""
        return self._engine

    @property
    def session_id(self) -> str | None:
        """Session ID — discovered or provided."""
        return self._session_id or self._engine.session_id

    @property
    def router(self) -> Router:
        """The output router."""
        return self._router

    @property
    def queue(self) -> MessageQueue:
        """The input queue."""
        return self._queue

    async def start(self, session_id: str | None = None) -> None:
        """Start the session — engine, replay (if resuming), and driver.

        Args:
            session_id: If provided, replay the session history and resume.
                        If None, start a new session.

        Session ID is not known until the first turn completes (for new sessions).
        Read it from session.session_id after the first ResultEvent.
        """
        self._session_id = session_id

        # Replay history if resuming
        if session_id:
            await self._replay(session_id)

        # Start the engine
        await self._engine.start(session_id=session_id)

        # Start the background driver that pulls from queue → engine → router
        self._driver_task = asyncio.create_task(self._drive())

    async def send(self, content: str, producer: str = "") -> None:
        """Send a message to the session.

        This queues the message. The driver will send it to the engine
        when it's ready.

        Args:
            content: The message text.
            producer: Name of the producer (for logging/debugging).
        """
        await self._queue.put(Message(content=content, producer=producer))

    async def events(self) -> AsyncIterator[Event]:
        """Consumer-facing event stream.

        Yields events from the router's consumer queue.
        Stops after a ResultEvent (end of turn).
        """
        async for event in self._router.events():
            yield event

    async def stop(self) -> None:
        """Shut down the session — queue, driver, engine."""
        # Close the queue to signal the driver to stop
        await self._queue.close()

        # Wait for the driver to finish
        if self._driver_task and not self._driver_task.done():
            try:
                await asyncio.wait_for(self._driver_task, timeout=5)
            except asyncio.TimeoutError:
                self._driver_task.cancel()
                try:
                    await self._driver_task
                except asyncio.CancelledError:
                    pass

        # Stop the engine
        await self._engine.stop()

        # Signal the consumer that events are done
        await self._router.close()

    async def _replay(self, session_id: str) -> None:
        """Replay session history through the router.

        Reads the JSONL transcript and routes replay events
        (with is_replay=True) so consumers can render history.
        """
        path = find_session_path(session_id)
        if path is None:
            return  # No history to replay — new session or missing file

        async for event in replay_session(path):
            await self._router.route_event(event)

    async def _drive(self) -> None:
        """Background driver: queue → engine → router.

        Pulls messages from the queue, sends them to the engine,
        and routes the engine's response events through the router.
        Runs until the queue is closed and drained.
        """
        while True:
            msg = await self._queue.get()
            if msg is None:
                return  # Queue closed — shut down

            if self._engine.state != EngineState.READY:
                return  # Engine died — stop driving

            # Send to engine
            await self._engine.send(msg.content)

            # Route all response events
            async for event in self._engine.events():
                await self._router.route_event(event)

                # Update session_id from ResultEvent if not yet known
                if isinstance(event, ResultEvent) and not self._session_id:
                    if event.session_id:
                        self._session_id = event.session_id
