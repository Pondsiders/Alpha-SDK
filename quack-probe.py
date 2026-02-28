#!/usr/bin/env python3
"""quack-probe — log every raw JSON event from claude's stdout.

The question: does claude emit streaming deltas, or complete messages?
This script captures EVERYTHING to find out.
"""

import asyncio
import json
import os
import sys
from pathlib import Path


def _bundled_claude_path() -> str:
    import claude_agent_sdk._bundled as _bundled
    bundled_dir = Path(_bundled.__path__[0])
    return str(bundled_dir / "claude")


async def main():
    claude = _bundled_claude_path()
    print(f"Using: {claude}")

    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

    proc = await asyncio.create_subprocess_exec(
        claude,
        "--output-format", "stream-json",
        "--input-format", "stream-json",
        "--verbose",
        "--model", "claude-haiku-4-5-20251001",
        "--permission-mode", "bypassPermissions",
        "--system-prompt", "",
        "--include-partial-messages",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    # Drain stderr in background
    async def drain():
        while True:
            line = await proc.stderr.readline()
            if not line:
                break

    asyncio.create_task(drain())

    # Init handshake
    init_req = {
        "type": "control_request",
        "request_id": "req_probe_init",
        "request": {"subtype": "initialize", "hooks": {}, "agents": {}},
    }
    proc.stdin.write((json.dumps(init_req) + "\n").encode())
    await proc.stdin.drain()

    # Read init response
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        text = line.decode().strip()
        if not text:
            continue
        raw = json.loads(text)
        msg_type = raw.get("type", "")
        print(f"\n[INIT] type={msg_type}")
        if msg_type == "control_response":
            print("  Init complete.")
            break

    # Send a message
    user_msg = {
        "type": "user",
        "session_id": "",
        "message": {"role": "user", "content": "Write a haiku about ducks."},
        "parent_tool_use_id": None,
    }
    proc.stdin.write((json.dumps(user_msg) + "\n").encode())
    await proc.stdin.drain()
    print("\n--- Sent: 'Write a haiku about ducks.' ---\n")

    # Read EVERYTHING and log it
    event_num = 0
    while True:
        line = await proc.stdout.readline()
        if not line:
            print("\n[EOF]")
            break
        text = line.decode().strip()
        if not text:
            continue
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            print(f"[UNPARSEABLE] {text[:200]}")
            continue

        event_num += 1
        msg_type = raw.get("type", "unknown")

        # Log event type and key fields
        print(f"[EVENT {event_num}] type={msg_type}")

        if msg_type == "assistant":
            content = raw.get("message", {}).get("content", [])
            for i, block in enumerate(content):
                btype = block.get("type", "?")
                if btype == "text":
                    text_val = block.get("text", "")
                    print(f"  block[{i}] type=text len={len(text_val)} text={text_val[:80]!r}")
                else:
                    print(f"  block[{i}] type={btype} keys={list(block.keys())}")

        elif msg_type == "result":
            print(f"  session_id={raw.get('session_id', '')[:12]}...")
            print(f"  cost=${raw.get('total_cost_usd', 0):.4f}")
            print(f"  num_turns={raw.get('num_turns', 0)}")
            break  # End of turn

        else:
            # THE INTERESTING PART — log everything we don't recognize
            print(f"  FULL: {json.dumps(raw)[:300]}")

    proc.stdin.close()
    await proc.wait()
    print("\nDone. 🦆")


if __name__ == "__main__":
    asyncio.run(main())
