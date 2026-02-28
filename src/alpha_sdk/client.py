"""client.py — AlphaClient. Thin composer of session from parts.

The client is where consumers configure the SDK. It composes an engine,
session, queue, and router from the parts the consumer provides, then
exposes the session's API.

This is the mirepoix: a generic foundation that doesn't taste like anyone.
Alpha adds soul, memory, orientation on top. Clyde uses it bare.

Usage:
    # Minimal — just engine + human input
    client = AlphaClient(model="claude-haiku-4-20250514")

    # With observers
    client = AlphaClient(
        model="claude-opus-4-6",
        observers=[suggest, archive],
    )

    # Full Alpha setup
    client = AlphaClient(
        model="claude-opus-4-6",
        system_prompt=soul,
        mcp_config="path/to/mcp.json",
        observers=[suggest, archive, broadcast],
    )

    session_id = await client.start()
    await client.send("Hello!")
    async for event in client.events():
        ...
    await client.stop()
"""

from __future__ import annotations

from typing import AsyncIterator

from .engine import CompactConfig, Engine, Event
from .queue import MessageQueue
from .router import Observer, Router
from .session import Session


class AlphaClient:
    """Composes an SDK session from parts.

    The client is the top-level entry point for consumers. It creates
    the engine, queue, router, and session, then delegates to the session
    for all operations.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        system_prompt: str | None = None,
        mcp_config: str | None = None,
        permission_mode: str = "bypassPermissions",
        observers: list[Observer] | None = None,
        compact_config: CompactConfig | None = None,
        extra_args: list[str] | None = None,
        queue_maxsize: int = 0,
    ):
        self._engine = Engine(
            model=model,
            system_prompt=system_prompt,
            mcp_config=mcp_config,
            permission_mode=permission_mode,
            extra_args=extra_args,
            compact_config=compact_config,
        )
        self._router = Router(observers=observers)
        self._queue = MessageQueue(maxsize=queue_maxsize)
        self._session = Session(
            engine=self._engine,
            router=self._router,
            queue=self._queue,
        )

    @property
    def session(self) -> Session:
        """The underlying session."""
        return self._session

    @property
    def engine(self) -> Engine:
        """The underlying engine."""
        return self._engine

    @property
    def session_id(self) -> str | None:
        """Current session ID."""
        return self._session.session_id

    async def start(self, session_id: str | None = None) -> None:
        """Start the client — engine, replay, driver.

        Args:
            session_id: Resume this session, or None for a new session.

        Session ID arrives in the event stream. Read it from
        client.session_id after the first turn.
        """
        await self._session.start(session_id=session_id)

    async def send(self, content: str, producer: str = "") -> None:
        """Send a message to the session."""
        await self._session.send(content, producer=producer)

    async def events(self) -> AsyncIterator[Event]:
        """Consumer-facing event stream."""
        async for event in self._session.events():
            yield event

    async def stop(self) -> None:
        """Shut down everything."""
        await self._session.stop()
