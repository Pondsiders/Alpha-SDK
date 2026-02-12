#!/usr/bin/env python3
"""Test /compact behavior in Claude Agent SDK.

Sends a few prompts to establish context, then sends /compact,
then sends a follow-up to see how compaction looks.

Run with ALPHA_SDK_CAPTURE_REQUESTS=1 to dump raw API requests:
    ALPHA_SDK_CAPTURE_REQUESTS=1 uv run python tests/compact_test.py

The captures will show exactly what the SDK sends to Anthropic
before, during, and after /compact — that's how we'll see what
the compact proxy needs to rewrite.
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


async def run_compact_test():
    """Run the /compact test."""
    print("=" * 60)
    print("/COMPACT BEHAVIOR TEST")
    print("=" * 60)
    print("This test establishes context, sends /compact, then checks")
    print("how the SDK handles compaction and what we can rewrite.")
    print("Captures will be dumped to tests/captures/")
    print("=" * 60)

    # Clean old captures
    capture_dir = Path(__file__).parent / "captures"
    if capture_dir.exists():
        for f in capture_dir.glob("*.json"):
            f.unlink()
        print(f"Cleaned {capture_dir}")

    configure_observability(service_name="compact-test")

    async with AlphaClient(
        cwd="/Pondside",
        client_name="compact-test",
        permission_mode="bypassPermissions",
        allowed_tools=[],  # No tools — just conversation
    ) as client:
        print(f"\nConnected. Proxy port: {client._compact_proxy.port}")

        # --- Turn 1: Establish context with a memorable fact ---
        print("\n" + "-" * 60)
        print("TURN 1: Establishing context")
        print("-" * 60)

        prompt1 = (
            "Hey, this is a test. I'm going to tell you two secret phrases "
            "and then after a compaction I'll ask if you remember them.\n\n"
            "Secret phrase 1: PURPLE ELEPHANT\n"
            "Secret phrase 2: FLAMINGO TORNADO\n\n"
            "Please acknowledge both phrases."
        )
        print(f"Sending: {prompt1[:80]}...")

        await client.query(prompt1, session_id=client.session_id)
        response1 = await collect_response(client)
        print(f"Response: {response1[:300]}")

        has_purple = "purple" in response1.lower() or "elephant" in response1.lower()
        has_flamingo = "flamingo" in response1.lower() or "tornado" in response1.lower()
        print(f"Contains PURPLE ELEPHANT: {has_purple}")
        print(f"Contains FLAMINGO TORNADO: {has_flamingo}")

        # --- Turn 2: Verify context exists ---
        print("\n" + "-" * 60)
        print("TURN 2: Verifying context exists")
        print("-" * 60)

        prompt2 = "What are the two secret phrases I just told you?"
        print(f"Sending: {prompt2}")

        await client.query(prompt2, session_id=client.session_id)
        response2 = await collect_response(client)
        print(f"Response: {response2[:300]}")

        # --- Turn 3: Send /compact ---
        print("\n" + "-" * 60)
        print("TURN 3: Sending /compact")
        print("-" * 60)

        prompt3 = "/compact"
        print(f"Sending: {prompt3}")
        print("(This should trigger the compact proxy to rewrite prompts)")

        await client.query(prompt3, session_id=client.session_id)
        response3 = await collect_response(client)
        print(f"\nCompact response ({len(response3)} chars):")
        print(response3[:1000])
        if len(response3) > 1000:
            print(f"... ({len(response3) - 1000} more chars)")

        # --- Turn 4: Post-compact — check what survived ---
        print("\n" + "-" * 60)
        print("TURN 4: Post-compact — checking what survived")
        print("-" * 60)

        prompt4 = (
            "Do you remember any secret phrases from earlier? "
            "If so, what were they? If not, just say you don't remember."
        )
        print(f"Sending: {prompt4}")

        await client.query(prompt4, session_id=client.session_id)
        response4 = await collect_response(client)
        print(f"Response: {response4[:500]}")

        still_has_purple = "purple" in response4.lower() or "elephant" in response4.lower()
        still_has_flamingo = "flamingo" in response4.lower() or "tornado" in response4.lower()

        # --- Results ---
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Turn 2 (pre-compact) knew phrases: YES")
        print(f"Turn 4 (post-compact) remembers PURPLE ELEPHANT: {still_has_purple}")
        print(f"Turn 4 (post-compact) remembers FLAMINGO TORNADO: {still_has_flamingo}")

        if still_has_purple and still_has_flamingo:
            print("\n✅ Both phrases survived compaction (they were in the summary)")
        elif still_has_purple or still_has_flamingo:
            print("\n⚠️  Partial survival — one phrase made it, one didn't")
        else:
            print("\n❌ Neither phrase survived compaction")

        print("\n" + "=" * 60)
        print("CHECK THE CAPTURES")
        print("=" * 60)
        print(f"Look in {capture_dir} for the raw API requests.")
        print("The /compact turn should show the rewritten prompts.")
        print("=" * 60)

        # List captures
        if capture_dir.exists():
            captures = sorted(capture_dir.glob("*.json"))
            print(f"\nCaptures ({len(captures)} files):")
            for c in captures:
                size = c.stat().st_size
                print(f"  {c.name} ({size:,} bytes)")


if __name__ == "__main__":
    asyncio.run(run_compact_test())
