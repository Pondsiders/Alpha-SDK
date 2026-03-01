#!/usr/bin/env python3
"""Probe: Does `claude --system-prompt` accept JSON arrays?

Simple test — spawn claude, see if it lives or dies with JSON.
Then capture the API request to see what it sends.
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import Request, urlopen
import threading


def start_capture_proxy(port: int, capture_file: str):
    """Start a minimal capture proxy in a background thread."""
    captured = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body)
                system = data.get("system")
                captured.append({"system": system, "path": self.path})
                with open(capture_file, "w") as f:
                    json.dump(captured, f, indent=2)
            except json.JSONDecodeError:
                pass

            # Forward to real Anthropic
            headers = {}
            for k, v in self.headers.items():
                if k.lower() not in ("host", "content-length", "transfer-encoding"):
                    headers[k] = v
            headers["Content-Length"] = str(len(body))

            req = Request(
                f"https://api.anthropic.com{self.path}",
                data=body,
                headers=headers,
                method="POST",
            )
            try:
                resp = urlopen(req)
                self.send_response(resp.status)
                for k, v in resp.getheaders():
                    if k.lower() not in ("transfer-encoding",):
                        self.send_header(k, v)
                self.end_headers()
                self.wfile.write(resp.read())
            except Exception as e:
                self.send_response(502)
                self.end_headers()
                self.wfile.write(str(e).encode())

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


async def run_one(name: str, system_prompt_arg: str, port: int, capture_file: str):
    """Run one experiment."""
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"  --system-prompt length: {len(system_prompt_arg)} chars")
    print(f"  first 200 chars: {system_prompt_arg[:200]}")
    print(f"{'='*60}")

    # Clean up old capture
    if os.path.exists(capture_file):
        os.unlink(capture_file)

    # Start proxy
    server = start_capture_proxy(port, capture_file)

    await asyncio.sleep(0.5)

    # Spawn claude
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}"
    # Clear nested session guard
    env.pop("CLAUDECODE", None)

    proc = await asyncio.create_subprocess_exec(
        "claude",
        "--output-format", "stream-json",
        "--input-format", "stream-json",
        "--verbose",
        "--model", "haiku",
        "--system-prompt", system_prompt_arg,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    # Wait for init
    await asyncio.sleep(3)

    # Check if process died
    if proc.returncode is not None:
        stderr = await proc.stderr.read()
        print(f"  CLAUDE DIED with code {proc.returncode}")
        print(f"  stderr: {stderr.decode()[:500]}")
        server.shutdown()
        return None

    # Send message
    msg = json.dumps({
        "type": "user",
        "message": {"role": "user", "content": "Say 'hello' and nothing else."},
    }) + "\n"

    proc.stdin.write(msg.encode())
    await proc.stdin.drain()

    # Collect output for a bit
    await asyncio.sleep(8)

    # Kill
    proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=3)
    except asyncio.TimeoutError:
        proc.kill()

    server.shutdown()

    # Read capture
    if os.path.exists(capture_file):
        with open(capture_file) as f:
            captures = json.load(f)

        # Find the /v1/messages request (not count_tokens)
        for cap in captures:
            if "/v1/messages" in cap.get("path", "") and "count" not in cap.get("path", ""):
                system = cap["system"]
                print(f"\n  API request system field:")
                if system is None:
                    print(f"    None")
                elif isinstance(system, str):
                    print(f"    TYPE: string")
                    print(f"    LENGTH: {len(system)}")
                    print(f"    CONTENT: {system[:300]}")
                elif isinstance(system, list):
                    print(f"    TYPE: array of {len(system)} blocks")
                    for j, block in enumerate(system):
                        text = block.get("text", "")
                        cc = block.get("cache_control")
                        print(f"    [{j}] type={block.get('type')} len={len(text)} cc={cc}")
                        print(f"        {text[:100]}")
                return captures
    else:
        print("  No capture file!")

    return None


async def main():
    # Experiment 1: JSON array
    result_json = await run_one(
        "JSON array",
        json.dumps([
            {"type": "text", "text": "You are a test assistant. Block one."},
            {"type": "text", "text": "Additional instructions. Block two."},
        ]),
        port=18997,
        capture_file="/tmp/probe_json.json",
    )

    # Experiment 2: Plain string
    result_plain = await run_one(
        "Plain string",
        "You are a test assistant. Block one.\n\nAdditional instructions. Block two.",
        port=18998,
        capture_file="/tmp/probe_plain.json",
    )

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"JSON array:  {'captured' if result_json else 'FAILED'}")
    print(f"Plain string: {'captured' if result_plain else 'FAILED'}")


if __name__ == "__main__":
    asyncio.run(main())
