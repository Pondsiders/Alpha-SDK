"""SessionBroadcast — fan-out event distribution with sequence tracking.

Replaces the singleton asyncio.Queue pattern in AlphaClient.
Multiple consumers each get their own subscriber queue — two browsers
both see every event instead of splitting them randomly.

Sequence numbers enable reconnection: pass Last-Event-ID from the SSE
spec to subscribe(from_seq=N) and get a replay of buffered events.

This is the fix for Duckpond #21 ("Two browsers, one stream").
"""

import asyncio
from collections import deque
from typing import Any


class SessionBroadcast:
    """Fan-out event distribution with sequence tracking.

    Events get monotonic sequence numbers and land in a ring buffer.
    Subscribers receive all events published after they subscribe,
    plus a replay of buffered events for late-joiners or reconnectors.

    Usage:
        broadcast = SessionBroadcast()

        # Producer side (AlphaClient._push_event)
        broadcast.publish("text-delta", {"text": "Hello"})

        # Consumer side (one per SSE connection)
        queue = broadcast.subscribe()
        event = await queue.get()  # {"type": "text-delta", "data": {...}, "id": 1}
        broadcast.unsubscribe(queue)

        # Reconnection (browser was backgrounded, comes back)
        queue = broadcast.subscribe(from_seq=last_event_id)
        # ^ replays everything it missed from the ring buffer
    """

    DEFAULT_BUFFER_SIZE = 1000

    def __init__(self, buffer_size: int = DEFAULT_BUFFER_SIZE):
        self._buffer: deque[dict[str, Any]] = deque(maxlen=buffer_size)
        self._seq: int = 0
        self._subscribers: set[asyncio.Queue] = set()
        self._closed: bool = False

    @property
    def seq(self) -> int:
        """Current sequence number (last assigned)."""
        return self._seq

    @property
    def subscriber_count(self) -> int:
        """Number of active subscribers."""
        return len(self._subscribers)

    def publish(self, event_type: str, data: dict[str, Any] | None = None) -> int:
        """Publish an event to all subscribers.

        Assigns a monotonic sequence number. Sync — safe to call from
        callbacks (e.g., CompactProxy token count updates).

        Args:
            event_type: SSE event type (e.g., "text-delta", "turn-end")
            data: Event payload

        Returns:
            The assigned sequence number.
        """
        if self._closed:
            return self._seq

        self._seq += 1
        event: dict[str, Any] = {
            "type": event_type,
            "data": data or {},
            "id": self._seq,
        }

        # Buffer for replay on reconnection
        self._buffer.append(event)

        # Fan out to all current subscribers
        for queue in self._subscribers:
            queue.put_nowait(event)

        return self._seq

    def subscribe(self, from_seq: int = 0) -> asyncio.Queue:
        """Create a subscriber queue.

        If from_seq > 0, replays buffered events with id > from_seq.
        This enables reconnection: pass Last-Event-ID from the SSE spec.

        Args:
            from_seq: Replay events after this sequence number (0 = no replay)

        Returns:
            An asyncio.Queue that receives all future events.
            None sentinel signals shutdown.
        """
        queue: asyncio.Queue = asyncio.Queue()

        # Replay buffered events for reconnection
        if from_seq > 0:
            for event in self._buffer:
                if event["id"] > from_seq:
                    queue.put_nowait(event)

        self._subscribers.add(queue)

        # If already closed, immediately signal
        if self._closed:
            queue.put_nowait(None)

        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Remove a subscriber.

        Safe to call even if the queue isn't subscribed (idempotent).
        """
        self._subscribers.discard(queue)

    def close(self) -> None:
        """Signal all subscribers to stop.

        Sends None sentinel to every subscriber queue.
        New subscribers after close() immediately receive None.
        Idempotent — safe to call multiple times.
        """
        if self._closed:
            return
        self._closed = True
        for queue in self._subscribers:
            queue.put_nowait(None)
