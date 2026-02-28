"""producers — Things that put messages on the queue.

Each producer is an async function or class that pushes messages
to the session's queue. The producer protocol is simple:

    async def run(self, session: Session) -> None:
        while should_continue:
            content = await get_input_somehow()
            await session.send(content, producer="my-name")

Producers are started by the client and run concurrently.
"""

from .human import HumanInput

__all__ = ["HumanInput"]
