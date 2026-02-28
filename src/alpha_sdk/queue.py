"""queue.py — Async message queue. Multiple producers, one consumer.

Producers call put() to enqueue messages. The session driver calls
get() to pull the next message for the engine. That's the whole API.

The queue supports:
- Multiple concurrent producers (each awaits put())
- One consumer (the session driver)
- Graceful shutdown via close() — drains remaining messages
- Backpressure via bounded size (producers block when full)

Messages are plain strings. The session driver handles formatting
them as JSON for the engine.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field


@dataclass
class Message:
    """A message on the queue.

    content: The text content to send to the engine.
    producer: Name of the producer that created this message (for logging/debugging).
    """

    content: str
    producer: str = ""


class MessageQueue:
    """Async bounded queue for producer → session driver communication.

    Multiple producers put messages. One consumer (the session driver)
    gets them. Close signals no more messages are coming.
    """

    def __init__(self, maxsize: int = 0):
        """Create a message queue.

        Args:
            maxsize: Maximum queue size. 0 = unbounded.
                     When full, put() blocks until space is available.
        """
        self._queue: asyncio.Queue[Message | None] = asyncio.Queue(maxsize=maxsize)
        self._closed = False

    @property
    def closed(self) -> bool:
        """Whether the queue has been closed for new messages."""
        return self._closed

    @property
    def qsize(self) -> int:
        """Current number of messages in the queue."""
        return self._queue.qsize()

    @property
    def empty(self) -> bool:
        """Whether the queue is empty."""
        return self._queue.empty()

    async def put(self, message: Message) -> None:
        """Add a message to the queue.

        Blocks if the queue is full (backpressure).
        Raises RuntimeError if the queue is closed.
        """
        if self._closed:
            raise RuntimeError("Cannot put to a closed queue")
        await self._queue.put(message)

    def put_nowait(self, message: Message) -> None:
        """Add a message without waiting.

        Raises asyncio.QueueFull if the queue is at capacity.
        Raises RuntimeError if the queue is closed.
        """
        if self._closed:
            raise RuntimeError("Cannot put to a closed queue")
        self._queue.put_nowait(message)

    async def get(self) -> Message | None:
        """Get the next message from the queue.

        Returns None when the queue is closed AND drained.
        This is the shutdown signal for the consumer.
        """
        msg = await self._queue.get()
        return msg  # None = sentinel

    def get_nowait(self) -> Message | None:
        """Get a message without waiting.

        Raises asyncio.QueueEmpty if no messages are available.
        """
        return self._queue.get_nowait()

    async def close(self) -> None:
        """Close the queue for new messages and signal the consumer.

        After close():
        - put() raises RuntimeError
        - The consumer will receive remaining messages, then None (sentinel)
        """
        if self._closed:
            return
        self._closed = True
        await self._queue.put(None)  # Sentinel — tells consumer we're done

    async def drain(self) -> list[Message]:
        """Drain all remaining messages without blocking.

        Returns messages that were in the queue. Does not include
        the sentinel. Useful for cleanup/testing.
        """
        messages = []
        while not self._queue.empty():
            try:
                msg = self._queue.get_nowait()
                if msg is not None:
                    messages.append(msg)
            except asyncio.QueueEmpty:
                break
        return messages
