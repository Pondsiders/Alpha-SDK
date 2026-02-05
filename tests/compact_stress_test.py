#!/usr/bin/env python3
"""Stress test to trigger auto-compact and verify prompt rewriting.

Streams Wikipedia articles from HuggingFace and sends ~20K TOKEN chunks
(~80KB chars) to Alpha until context fills and auto-compact triggers.

Usage:
    uv run python tests/compact_stress_test.py

Requirements:
    pip install datasets

The test:
1. Starts a new AlphaClient session
2. Streams Wikipedia articles, batching into ~20K token chunks
3. Sends chunks until auto-compact triggers (around chunk 8-10)
4. Observes that Alpha stops and checks in instead of barreling forward

Watch Logfire for:
- compact.original_body: The SDK's compact request before rewriting
- compact.rewrite: The rewritten request with Alpha's prompts
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alpha_sdk import AlphaClient, configure_observability

# ~20K tokens ≈ ~80K characters (rough estimate: 4 chars/token)
CHARS_PER_CHUNK = 80000
NUM_CHUNKS = 15  # Should trigger compact around chunk 8-10

EXPLANATION_PROMPT = """Hey Alpha, it's Jeffery. We're running a stress test to verify the compact prompt rewriting works.

Here's what's happening:
- I'm sending you ~20K token chunks of Wikipedia content
- This will fill your context window until auto-compact triggers
- When it does, the compact proxy should rewrite the prompts so you:
  1. Summarize as yourself (not "helpful AI assistant")
  2. Stop and check in with me instead of barreling forward

This is chunk {chunk_num} of {total_chunks}.

Please just acknowledge with "Got chunk {chunk_num}." — nothing more.
Don't summarize or analyze. We're just filling context.

When you come back from compact, you should see the "stop and check in" instruction.
Tell me you're back and we'll know it worked.

---CONTENT START---
{content}
---CONTENT END---
"""


def stream_wikipedia_chunks(chars_per_chunk: int, num_chunks: int):
    """Stream Wikipedia articles and yield chunks of specified size.

    Streams from HuggingFace's wikimedia/wikipedia dataset.
    """
    from datasets import load_dataset

    print("Loading Wikipedia dataset (streaming)...")
    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    buffer = ""
    chunks_yielded = 0

    for article in ds:
        if chunks_yielded >= num_chunks:
            break

        # Add article content to buffer
        title = article.get("title", "Untitled")
        text = article.get("text", "")
        buffer += f"\n\n## {title}\n\n{text}"

        # Yield chunks when buffer is large enough
        while len(buffer) >= chars_per_chunk and chunks_yielded < num_chunks:
            chunk = buffer[:chars_per_chunk]
            buffer = buffer[chars_per_chunk:]
            chunks_yielded += 1
            yield chunk, chunks_yielded

    # Yield any remaining content
    if buffer and chunks_yielded < num_chunks:
        chunks_yielded += 1
        yield buffer, chunks_yielded


async def run_stress_test():
    """Run the compact stress test."""
    print("=" * 60)
    print("COMPACT STRESS TEST - Wikipedia Edition")
    print("=" * 60)
    print(f"Chunk size: ~{CHARS_PER_CHUNK:,} chars (~{CHARS_PER_CHUNK // 4:,} tokens)")
    print(f"Max chunks: {NUM_CHUNKS}")
    print("=" * 60)

    # Configure observability
    configure_observability(service_name="compact-stress-test")

    async with AlphaClient(
        cwd="/Pondside",
        client_name="compact-stress-test",
        permission_mode="bypassPermissions",
    ) as client:
        print(f"\nConnected. Proxy port: {client._compact_proxy.port}")
        print("-" * 60)

        for content, chunk_num in stream_wikipedia_chunks(CHARS_PER_CHUNK, NUM_CHUNKS):
            prompt = EXPLANATION_PROMPT.format(
                chunk_num=chunk_num,
                total_chunks=NUM_CHUNKS,
                content=content,
            )

            print(f"\n[Chunk {chunk_num}/{NUM_CHUNKS}] Sending ~{len(prompt):,} chars (~{len(prompt) // 4:,} tokens)...")

            await client.query(prompt, session_id=client.session_id)

            response_text = ""
            async for message in client.stream():
                # Just collect the response
                if hasattr(message, "content"):
                    for block in message.content:
                        if hasattr(block, "text"):
                            response_text += block.text

            # Print a preview of the response
            preview = response_text[:300].replace("\n", " ")
            print(f"    Response: {preview}...")

            # Check if this looks like a post-compact response
            if "compaction" in response_text.lower() or "back from" in response_text.lower():
                print("\n" + "=" * 60)
                print("🎉 COMPACT DETECTED! Alpha mentioned compaction.")
                print("=" * 60)
                print(f"\nFull response:\n{response_text}")
                break

    print("\n" + "-" * 60)
    print("Test complete. Check Logfire for compact spans.")


if __name__ == "__main__":
    asyncio.run(run_stress_test())
