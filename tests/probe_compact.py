"""probe_compact.py — Quick probe: does /compact work via Engine.send()?

NOT a test. A discovery script. Run it, watch what happens.

Usage: uv run python tests/probe_compact.py
"""

import asyncio
import os
from pathlib import Path

# Ensure we can capture what flows through the proxy
os.environ["ALPHA_SDK_CAPTURE_REQUESTS"] = "1"

from alpha_sdk.engine import (
    Engine,
    AssistantEvent,
    ResultEvent,
    SystemEvent,
    ErrorEvent,
    CompactConfig,
)

# Minimal flags — no tools, no browser, but KEEP slash commands enabled
# because /compact IS a slash command and we need it!
SECURE_TEST_ARGS = [
    "--tools", "",
    "--no-chrome",
    "--strict-mcp-config",
]

# Known markers so we can see rewriting in the captures
COMPACT_CONFIG = CompactConfig(
    system="PROBE_SYSTEM_REPLACEMENT",
    prompt="PROBE_COMPACT_INSTRUCTIONS_REPLACEMENT",
    continuation="PROBE_CONTINUATION_REPLACEMENT",
)

CAPTURE_DIR = Path(__file__).parent / "captures"


async def main():
    engine = Engine(
        model="haiku",
        system_prompt="Reply with exactly one short sentence. No tools.",
        compact_config=COMPACT_CONFIG,
        extra_args=SECURE_TEST_ARGS,
    )

    try:
        print("Starting engine...")
        await engine.start()
        print(f"Engine started. Model: {engine.model}")
        print(f"Proxy base URL: {engine._proxy.base_url}")
        print(f"Captures will land in: {CAPTURE_DIR}")
        print()

        # Hammer it with BIG messages to build real context
        # ~2K tokens per message × 8 turns ≈ 16K+ tokens
        filler = "The quick brown fox jumps over the lazy dog. " * 100  # ~2K tokens
        num_turns = 8
        for i in range(1, num_turns + 1):
            print(f"--- Turn {i}/{num_turns}: Building context ---")
            msg = f"This is test message #{i}. Here is filler text: {filler}\nReply with one sentence acknowledging message #{i}."
            await engine.send(msg)
            async for event in engine.events():
                if isinstance(event, AssistantEvent):
                    text = event.text
                    if text:
                        print(f"  Assistant: {text[:100]}")
                elif isinstance(event, ResultEvent):
                    print(f"  Result: cost=${event.cost_usd:.4f} tokens~={engine.token_count}")
                    break
                elif isinstance(event, ErrorEvent):
                    print(f"  ERROR: {event.message}")
                    return

        # Clear captures from context-building turns
        if CAPTURE_DIR.exists():
            for f in CAPTURE_DIR.glob("*.json"):
                f.unlink()
        print()
        print("Cleared pre-compact captures.")

        print()
        print(f"--- Turn {num_turns + 1}: Sending /compact ---")
        print("Watching for: compact API calls through the proxy")
        print()

        await engine.send("/compact")

        events_seen = []
        async for event in engine.events():
            events_seen.append(event)
            if isinstance(event, AssistantEvent):
                text = event.text
                if text:
                    print(f"  Assistant: {text[:200]}")
            elif isinstance(event, SystemEvent):
                print(f"  System: subtype={event.subtype}")
                print(f"    raw keys: {list(event.raw.keys())}")
            elif isinstance(event, ResultEvent):
                print(f"  Result: session={event.session_id[:20]}... cost=${event.cost_usd:.4f}")
                print(f"  is_error={event.is_error}")
                break
            elif isinstance(event, ErrorEvent):
                print(f"  ERROR: {event.message}")
                break
            else:
                print(f"  Unknown: {type(event).__name__}")
                print(f"    raw: {str(event.raw)[:200]}")

        print()
        print(f"Total events from /compact turn: {len(events_seen)}")
        print(f"Event types: {[type(e).__name__ for e in events_seen]}")
        print()

        # Check captures
        if CAPTURE_DIR.exists():
            captures = sorted(CAPTURE_DIR.glob("*.json"))
            print(f"Captured {len(captures)} request(s) during compact:")
            for c in captures:
                size = c.stat().st_size
                print(f"  {c.name} ({size} bytes)")

                # Show a preview of what we caught
                import json
                with open(c) as f:
                    data = json.load(f)
                system = data.get("system", "")
                if isinstance(system, list):
                    for block in system:
                        if isinstance(block, dict):
                            text = block.get("text", "")
                            if "PROBE_" in text or "summariz" in text.lower() or "helpful" in text.lower():
                                print(f"    ** SYSTEM: {text[:120]}")
                elif isinstance(system, str):
                    if "PROBE_" in system or "summariz" in system.lower() or "helpful" in system.lower():
                        print(f"    ** SYSTEM: {system[:120]}")

                messages = data.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    content = last_msg.get("content", "")
                    if isinstance(content, str):
                        preview = content[:150]
                    elif isinstance(content, list):
                        preview = str([b.get("text", "")[:60] for b in content if isinstance(b, dict)])[:150]
                    else:
                        preview = str(content)[:150]
                    print(f"    last msg ({last_msg.get('role', '?')}): {preview}")
        else:
            print("No captures directory found.")

    finally:
        await engine.stop()
        print("\nEngine stopped.")


if __name__ == "__main__":
    asyncio.run(main())
