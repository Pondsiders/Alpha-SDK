#!/usr/bin/env python3
"""Test /compact with an argument — where does the arg end up?

Sends /compact rubber baby buggy bumpers and captures before/after
to see how the argument gets threaded through the SDK and proxy.

Run with:
    ALPHA_SDK_CAPTURE_REQUESTS=1 uv run python tests/compact_arg_test.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Enable request capture
os.environ["ALPHA_SDK_CAPTURE_REQUESTS"] = "1"

from alpha_sdk import AlphaClient, configure_observability


async def collect_response(client: AlphaClient) -> str:
    """Collect the full text response from a stream."""
    text_parts = []
    async for message in client.stream():
        if hasattr(message, "content"):
            for block in message.content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)
    return "".join(text_parts)


async def run_compact_arg_test():
    """Run the /compact-with-argument test."""
    print("=" * 60)
    print("/COMPACT ARGUMENT TEST")
    print("=" * 60)
    print("Testing: /compact rubber baby buggy bumpers")
    print("Goal: Find where the argument lands in the API request")
    print("Captures will be dumped to tests/captures/")
    print("=" * 60)

    # Clean old captures
    capture_dir = Path(__file__).parent / "captures"
    if capture_dir.exists():
        for f in capture_dir.glob("*.json"):
            f.unlink()
        print(f"Cleaned {capture_dir}")

    configure_observability(service_name="compact-arg-test")

    async with AlphaClient(
        cwd="/Pondside",
        client_name="compact-arg-test",
        permission_mode="bypassPermissions",
        allowed_tools=[],
    ) as client:
        print(f"\nConnected. Proxy port: {client._compact_proxy.port}")

        # --- Turn 1: Establish some context ---
        print("\n" + "-" * 60)
        print("TURN 1: Establishing context")
        print("-" * 60)

        prompt1 = (
            "This is a test. Remember two things:\n"
            "1. PURPLE ELEPHANT — very important, this is the main work\n"
            "2. FLAMINGO TORNADO — this was a side thing, already done\n\n"
            "Acknowledge both."
        )
        print(f"Sending: {prompt1[:80]}...")

        await client.query(prompt1, session_id=client.session_id)
        response1 = await collect_response(client)
        print(f"Response: {response1[:300]}")

        # --- Turn 2: Send /compact WITH ARGUMENT ---
        print("\n" + "-" * 60)
        print("TURN 2: Sending /compact rubber baby buggy bumpers")
        print("-" * 60)

        prompt2 = "/compact rubber baby buggy bumpers"
        print(f"Sending: {prompt2}")

        await client.query(prompt2, session_id=client.session_id)
        response2 = await collect_response(client)
        print(f"\nCompact response ({len(response2)} chars):")
        print(response2[:1000])

        # --- Results ---
        print("\n" + "=" * 60)
        print("CAPTURES")
        print("=" * 60)

        if capture_dir.exists():
            captures = sorted(capture_dir.glob("*.json"))
            print(f"\n{len(captures)} files captured:")
            for c in captures:
                size = c.stat().st_size
                print(f"  {c.name} ({size:,} bytes)")

            # Quick grep for our phrase
            print("\n" + "-" * 60)
            print("SEARCHING FOR 'rubber baby buggy bumpers' IN CAPTURES")
            print("-" * 60)
            for c in captures:
                content = c.read_text()
                if "rubber baby buggy bumpers" in content.lower():
                    # Find the line
                    for i, line in enumerate(content.split("\n")):
                        if "rubber baby buggy bumpers" in line.lower():
                            print(f"\n  FOUND in {c.name}, line {i+1}:")
                            # Show some context
                            lines = content.split("\n")
                            start = max(0, i - 2)
                            end = min(len(lines), i + 3)
                            for j in range(start, end):
                                marker = ">>>" if j == i else "   "
                                print(f"    {marker} {lines[j][:120]}")
                else:
                    print(f"  NOT in {c.name}")

        print("\n" + "=" * 60)
        print("DONE")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_compact_arg_test())
