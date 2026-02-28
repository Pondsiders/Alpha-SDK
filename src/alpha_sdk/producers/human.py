"""human.py — Wraps human input as a producer.

The simplest producer: reads from stdin (or a provided async function)
and pushes each line to the session queue. Used by CLI consumers like
quack-next.

Usage:
    # Default: reads from stdin
    human = HumanInput()
    await human.run(session)

    # Custom input source (for testing, web UI, etc.)
    human = HumanInput(input_fn=my_async_reader)
    await human.run(session)
"""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING, Callable, Awaitable

if TYPE_CHECKING:
    from ..session import Session


async def _stdin_reader() -> str | None:
    """Read a line from stdin asynchronously.

    Returns None on EOF (Ctrl-D).
    """
    loop = asyncio.get_event_loop()
    try:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            return None  # EOF
        return line.rstrip("\n")
    except (EOFError, KeyboardInterrupt):
        return None


class HumanInput:
    """Producer that reads human input and sends it to the session.

    By default reads from stdin. Can be configured with a custom
    input function for testing or alternative interfaces.
    """

    def __init__(
        self,
        input_fn: Callable[[], Awaitable[str | None]] | None = None,
        prompt: str = "> ",
        name: str = "human",
    ):
        """
        Args:
            input_fn: Async function that returns the next input string,
                      or None to signal EOF. Defaults to stdin reader.
            prompt: Prompt string to display before reading input.
            name: Producer name for message attribution.
        """
        self._input_fn = input_fn or _stdin_reader
        self._prompt = prompt
        self._name = name

    async def run(self, session: Session) -> None:
        """Run the input loop until EOF or the session stops.

        Reads input, sends to session, repeats.
        """
        while True:
            # Show prompt (only for stdin)
            if self._input_fn is _stdin_reader:
                print(self._prompt, end="", flush=True)

            content = await self._input_fn()
            if content is None:
                break  # EOF — producer is done

            if not content.strip():
                continue  # Skip empty lines

            await session.send(content, producer=self._name)
