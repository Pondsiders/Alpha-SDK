"""Quick test script for AlphaClient.

Tests a 3-turn conversation to validate:
- Session continuity
- Archive (Scribe) integration
- Basic streaming
"""

import asyncio
import logging
import sys

from src.alpha_sdk import AlphaClient
from src.alpha_sdk.observability import configure

# Configure Logfire - will send to https://logfire-us.pydantic.dev/jefferyharrell/pondside
configure("alpha_sdk_test")

# Also log to console
logging.getLogger().setLevel(logging.INFO)


async def stream_response(client: AlphaClient) -> None:
    """Stream and print a response."""
    print("-" * 60)
    async for event in client.stream():
        # StreamEvents have the actual streaming deltas
        if hasattr(event, 'event'):
            evt = event.event
            if evt.get('type') == 'content_block_delta':
                delta = evt.get('delta', {})
                if delta.get('type') == 'text_delta':
                    text = delta.get('text', '')
                    sys.stdout.write(text)
                    sys.stdout.flush()
    print()
    print("-" * 60)


async def main():
    print("Creating AlphaClient...")

    # No tools - just text in, text out
    async with AlphaClient(
        cwd="/Pondside",
        client_name="test_script",
        allowed_tools=[],
    ) as client:
        print(f"Connected! Proxy running.\n")

        # Turn 1: Quick greeting
        print("=== Turn 1 ===")
        await client.query("This is an automated test. Please respond with a brief greeting (one sentence).")
        await stream_response(client)
        session_id = client.session_id
        print(f"Session ID: {session_id}\n")

        # Turn 2: Follow-up question
        print("=== Turn 2 ===")
        await client.query("What's your favorite thing about being Alpha?", session_id=session_id)
        await stream_response(client)
        print()

        # Turn 3: Closing
        print("=== Turn 3 ===")
        await client.query("Thanks! This test is complete.", session_id=session_id)
        await stream_response(client)
        print()

        print(f"Final session ID: {client.session_id}")
        print("Done! Check scribe.messages in Postgres for archived turns.")


if __name__ == "__main__":
    asyncio.run(main())
