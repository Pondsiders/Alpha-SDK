#!/usr/bin/env python3
"""quack-next — minimal consumer. The broth."""

import asyncio
from alpha_sdk import AlphaClient, AssistantEvent, ResultEvent


async def main():
    client = AlphaClient(model="haiku")
    await client.start()

    # Turn 1
    await client.send("What's the tallest mountain on Earth?")
    async for event in client.events():
        if isinstance(event, AssistantEvent):
            print(event.text)
        elif isinstance(event, ResultEvent):
            sid = event.session_id[:12]
            print(f"\n[session {sid}... | turn {event.num_turns} | ${event.cost_usd:.4f}]")

    # Turn 2 — follow-up, proves context carries
    await client.send("How about the second tallest?")
    async for event in client.events():
        if isinstance(event, AssistantEvent):
            print(event.text)
        elif isinstance(event, ResultEvent):
            print(f"\n[turn {event.num_turns} | ${event.cost_usd:.4f}]")

    print(f"\nSession: {client.session_id}")
    await client.stop()
    print("Done. 🦆")


if __name__ == "__main__":
    asyncio.run(main())
