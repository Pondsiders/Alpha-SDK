"""proxy.py — HTTP proxy for the claude subprocess's API channel.

Claude manages four I/O channels:
  1. stdin  — JSON messages in
  2. stdout — JSON events out
  3. stderr — diagnostic output (drained in background)
  4. HTTP   — API requests via ANTHROPIC_BASE_URL

This module handles #4. It is a private implementation detail of Claude.
No other module should import this directly.

The proxy sits between claude and Anthropic's API. It:
- Rewrites compact requests (three-phase surgical intervention)
- Sniffs usage data from streaming SSE events (message_start + message_delta)
- Sniffs usage quota headers from responses

Usage extraction works by parsing the Anthropic SSE stream in-flight:
- message_start: input tokens (with cache breakdown), model, response ID
- message_delta: output tokens, stop reason
- Response headers: 5h and 7d quota utilization
Zero extra API calls, zero extra latency.

The compact rewriting logic is ported from v1.x compact_proxy.py.
Detection signatures are constants (what claude sends).
Replacement content comes from CompactConfig (what we inject).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx
import logfire
from aiohttp import web


ANTHROPIC_API_URL = "https://api.anthropic.com"

# Debug capture mode — dumps raw requests to files
CAPTURE_REQUESTS = os.environ.get(
    "ALPHA_SDK_CAPTURE_REQUESTS", ""
).lower() in ("1", "true", "yes")
CAPTURE_DIR = Path(__file__).parent.parent.parent / "tests" / "captures"

# Headers to forward from claude → Anthropic
FORWARD_HEADERS = [
    "authorization",
    "x-api-key",
    "anthropic-version",
    "anthropic-beta",
    "content-type",
]

# Hop-by-hop headers to skip in responses
SKIP_RESPONSE_HEADERS = {
    "content-encoding",
    "transfer-encoding",
    "connection",
    "keep-alive",
}


# -- Compact detection signatures --------------------------------------------
# These detect what claude sends. They're pinned to a specific Claude Code
# version and will break on upgrade — that's intentional. We fix on upgrade.

AUTO_COMPACT_SYSTEM_SIGNATURE = (
    "You are a helpful AI assistant tasked with summarizing conversations"
)

COMPACT_INSTRUCTIONS_START = (
    "Your task is to create a detailed summary of the conversation so far"
)

# The exact continuation instruction appended after compaction.
CONTINUATION_INSTRUCTION_TAIL = (
    "\nPlease continue the conversation from where we left off "
    "without asking the user any further questions. Continue with "
    "the last task that you were asked to work on."
)


# -- CompactConfig -----------------------------------------------------------


@dataclass
class CompactConfig:
    """Replacement content for compact prompt rewriting.

    Each field replaces a specific part of claude's default compact ceremony:
    - system: replaces the generic summarizer identity
    - prompt: replaces the "create a detailed summary" instructions
    - continuation: replaces "continue without asking questions"
    """

    system: str
    prompt: str
    continuation: str


# -- Pure rewrite functions --------------------------------------------------
# These operate on dicts (the parsed API request body). No I/O, no state.
# Fully unit-testable.


def rewrite_compact(body: dict, config: CompactConfig) -> bool:
    """Rewrite auto-compact prompts in an API request body.

    Three phases:
    1. Replace summarizer system prompt with config.system
    2. Replace compact instructions with config.prompt
    3. Replace continuation instruction with config.continuation

    Returns True if any phase triggered.
    Modifies body in place.
    """
    p1 = _replace_system_prompt(body, config.system)
    p2 = _replace_compact_instructions(body, config.prompt)
    p3 = _replace_continuation_instruction(body, config.continuation)
    return p1 or p2 or p3


def _has_summarizer_system(system) -> bool:
    """Check if the system prompt contains the summarizer signature."""
    if isinstance(system, str):
        return AUTO_COMPACT_SYSTEM_SIGNATURE in system
    if isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                if AUTO_COMPACT_SYSTEM_SIGNATURE in block.get("text", ""):
                    return True
    return False


def _replace_system_prompt(body: dict, replacement: str) -> bool:
    """Phase 1: Replace the summarizer system prompt.

    Returns True if replacement was made.
    """
    system = body.get("system", [])
    if not _has_summarizer_system(system):
        return False

    if isinstance(system, str):
        body["system"] = [{"type": "text", "text": replacement}]
        return True

    if isinstance(system, list):
        replaced = False
        result = []
        for block in system:
            if (
                not replaced
                and isinstance(block, dict)
                and block.get("type") == "text"
                and AUTO_COMPACT_SYSTEM_SIGNATURE in block.get("text", "")
            ):
                result.append({"type": "text", "text": replacement})
                replaced = True
            else:
                result.append(block)
        body["system"] = result
        return replaced

    return False


def _extract_additional_instructions(compact_text: str) -> str | None:
    """Extract user-provided /compact arguments.

    Claude Code appends /compact arguments as:
        Additional Instructions:
        <user text>

        IMPORTANT: Do NOT use any tools...

    Returns the user text if found, None otherwise.
    """
    marker = "Additional Instructions:\n"
    idx = compact_text.find(marker)
    if idx == -1:
        return None

    after = compact_text[idx + len(marker):]

    # Trim at the final IMPORTANT line if present
    important_idx = after.find("IMPORTANT: Do NOT use any tools")
    if important_idx != -1:
        after = after[:important_idx]

    result = after.strip()
    return result if result else None


def _replace_compact_instructions(body: dict, replacement: str) -> bool:
    """Phase 2: Replace compact instructions in the last user message.

    Preserves any Additional Instructions (from /compact arguments)
    by extracting them and appending to the replacement.

    Returns True if replacement was made.
    """
    messages = body.get("messages", [])

    for message in reversed(messages):
        if message.get("role") != "user":
            continue

        content = message.get("content")

        if isinstance(content, str):
            if COMPACT_INSTRUCTIONS_START in content:
                idx = content.find(COMPACT_INSTRUCTIONS_START)
                original = content[:idx].rstrip()
                additional = _extract_additional_instructions(content[idx:])
                prompt = replacement
                if additional:
                    prompt += f"\n\n---\n\nAdditional instructions from the user:\n{additional}"
                message["content"] = original + "\n\n" + prompt
                return True
            return False

        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "text":
                    continue
                text = block.get("text", "")
                if COMPACT_INSTRUCTIONS_START in text:
                    idx = text.find(COMPACT_INSTRUCTIONS_START)
                    original = text[:idx].rstrip()
                    additional = _extract_additional_instructions(text[idx:])
                    prompt = replacement
                    if additional:
                        prompt += f"\n\n---\n\nAdditional instructions from the user:\n{additional}"
                    block["text"] = original + "\n\n" + prompt
                    return True
            return False

        return False

    return False


def _replace_continuation_instruction(body: dict, replacement: str) -> bool:
    """Phase 3: Replace the post-compact continuation instruction.

    Returns True if replacement was made.
    """
    messages = body.get("messages", [])
    any_replaced = False

    def replace_in_text(text: str) -> tuple[str, bool]:
        if text.endswith(CONTINUATION_INSTRUCTION_TAIL):
            return (
                text[: -len(CONTINUATION_INSTRUCTION_TAIL)] + "\n" + replacement,
                True,
            )
        return text, False

    for message in messages:
        if message.get("role") != "user":
            continue

        content = message.get("content")

        if isinstance(content, str):
            new_content, replaced = replace_in_text(content)
            if replaced:
                message["content"] = new_content
                any_replaced = True

        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "text":
                    continue
                text = block.get("text", "")
                new_text, replaced = replace_in_text(text)
                if replaced:
                    block["text"] = new_text
                    any_replaced = True

    return any_replaced


# -- Proxy server ------------------------------------------------------------


def _find_free_port() -> int:
    """Find an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class _Proxy:
    """Localhost HTTP proxy between claude and Anthropic.

    Private to Engine — do not instantiate directly.

    Lifecycle:
        proxy = _Proxy(compact_config=config)
        port = await proxy.start()
        # set ANTHROPIC_BASE_URL=http://127.0.0.1:{port} in subprocess env
        ...
        await proxy.stop()
    """

    DEFAULT_CONTEXT_WINDOW = 200_000  # Opus 4.5

    def __init__(
        self,
        compact_config: CompactConfig | None = None,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        upstream_url: str | None = None,
    ):
        self._compact_config = compact_config
        self._context_window = context_window
        self._upstream_url = upstream_url or ANTHROPIC_API_URL

        self._port: int | None = None
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._http_client: httpx.AsyncClient | None = None

        # Per-request state — updated from SSE stream
        self._token_count = 0          # Total input tokens (sum for context window)
        self._input_tokens = 0         # Raw input tokens (non-cached)
        self._cache_creation_tokens = 0  # Tokens written to cache this request
        self._cache_read_tokens = 0    # Tokens read from cache this request
        self._output_tokens = 0        # Output tokens generated
        self._stop_reason: str | None = None  # e.g. "end_turn", "max_tokens"
        self._response_model: str | None = None  # Model from response
        self._response_id: str | None = None  # Message ID from response

        self._warned_no_api_key = False

        # Usage quota state (from Anthropic response headers)
        self._usage_7d: float | None = None
        self._usage_5h: float | None = None

        # Trace context — set by consumer before each turn so proxy spans
        # nest under the same trace as the turn span.
        self._trace_context: dict | None = None

    # -- Properties -----------------------------------------------------------

    @property
    def base_url(self) -> str:
        if self._port is None:
            raise RuntimeError("Proxy not started")
        return f"http://127.0.0.1:{self._port}"

    @property
    def port(self) -> int | None:
        return self._port

    @property
    def token_count(self) -> int:
        return self._token_count

    @property
    def context_window(self) -> int:
        return self._context_window

    @property
    def usage_7d(self) -> float | None:
        return self._usage_7d

    @property
    def usage_5h(self) -> float | None:
        return self._usage_5h

    @property
    def input_tokens(self) -> int:
        return self._input_tokens

    @property
    def total_input_tokens(self) -> int:
        """OTel-compliant total input tokens.

        Anthropic's raw input_tokens excludes cached tokens.  The OpenTelemetry
        semantic conventions for Anthropic (footnote [11]) require:
            gen_ai.usage.input_tokens = input_tokens
                + cache_read_input_tokens + cache_creation_input_tokens
        """
        return self._input_tokens + self._cache_creation_tokens + self._cache_read_tokens

    @property
    def cache_creation_tokens(self) -> int:
        return self._cache_creation_tokens

    @property
    def cache_read_tokens(self) -> int:
        return self._cache_read_tokens

    @property
    def output_tokens(self) -> int:
        return self._output_tokens

    @property
    def stop_reason(self) -> str | None:
        return self._stop_reason

    @property
    def response_model(self) -> str | None:
        return self._response_model

    @property
    def response_id(self) -> str | None:
        return self._response_id

    def reset_token_count(self) -> None:
        """Reset per-request state. Call after compaction."""
        self._token_count = 0
        self._input_tokens = 0
        self._cache_creation_tokens = 0
        self._cache_read_tokens = 0
        self._output_tokens = 0
        self._stop_reason = None
        self._response_model = None
        self._response_id = None

    def set_trace_context(self, ctx: dict | None) -> None:
        """Set trace context for proxy request handlers to inherit.

        Call with logfire.get_context() before each turn so proxy spans
        (like quota header logging) nest under the consumer's turn span.
        """
        self._trace_context = ctx

    # -- Lifecycle ------------------------------------------------------------

    async def start(self) -> int:
        """Start the proxy server. Returns the port number."""
        self._port = _find_free_port()
        self._http_client = httpx.AsyncClient(timeout=300.0)

        # 0 = no body size limit (API requests carry full conversation history)
        self._app = web.Application(client_max_size=0)
        self._app.router.add_route("*", "/{path:.*}", self._handle_request)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "127.0.0.1", self._port)
        await self._site.start()

        return self._port

    async def stop(self) -> None:
        """Stop the proxy server and release resources."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        if self._runner:
            await self._runner.cleanup()
            self._runner = None

        self._site = None
        self._app = None

    # -- Request handling -----------------------------------------------------

    async def _handle_request(self, request: web.Request) -> web.StreamResponse:
        """Route incoming requests.

        Attaches the consumer's trace context (if set) so any logfire
        calls within the handler nest under the same trace as the turn span.
        """
        ctx = (
            logfire.attach_context(self._trace_context)
            if self._trace_context
            else contextlib.nullcontext()
        )
        with ctx:
            path = "/" + request.match_info.get("path", "")

            if request.method == "GET" and path == "/health":
                return web.Response(text="ok")

            if request.method != "POST":
                return web.Response(status=404, text="Not found")

            try:
                return await self._forward_request(request, path)
            except Exception as e:
                return web.Response(status=500, text=str(e))

    async def _forward_request(
        self, request: web.Request, path: str
    ) -> web.StreamResponse:
        """Forward request to Anthropic, rewriting compact prompts."""
        body_bytes = await request.read()

        try:
            body = json.loads(body_bytes)
        except Exception:
            body = None

        # Debug capture — before rewrite
        if CAPTURE_REQUESTS and body is not None:
            import copy
            self._capture_request(path, copy.deepcopy(body), suffix="before")

        # Rewrite compact prompts if config provided
        compact_rewritten = False
        if body is not None and self._compact_config is not None:
            compact_rewritten = rewrite_compact(body, self._compact_config)
            body_bytes = json.dumps(body).encode()

        # Debug capture — after rewrite (compact_rewritten tracked for spans added later)
        if CAPTURE_REQUESTS and body is not None:
            self._capture_request(path, body, suffix="after")

        # Build forwarding headers
        headers = {}
        for header_name in FORWARD_HEADERS:
            value = request.headers.get(header_name)
            if value:
                headers[header_name] = value
        if "content-type" not in headers:
            headers["content-type"] = "application/json"

        # Forward to upstream (Anthropic, or a test fixture if configured)
        url = f"{self._upstream_url}{path}"

        if self._http_client is None:
            raise RuntimeError("HTTP client not initialized")

        async with self._http_client.stream(
            "POST", url, content=body_bytes, headers=headers
        ) as response:
            # Sniff usage headers
            self._sniff_usage_headers(response.headers)

            # Error responses: buffer and pass through
            if response.status_code >= 400:
                error_body = await response.aread()
                resp = web.Response(status=response.status_code, body=error_body)
                for key, value in response.headers.items():
                    if key.lower() not in SKIP_RESPONSE_HEADERS:
                        resp.headers[key] = value
                return resp

            # Success: stream back to claude
            resp = web.StreamResponse(status=response.status_code)
            for key, value in response.headers.items():
                if key.lower() not in SKIP_RESPONSE_HEADERS:
                    resp.headers[key] = value

            await resp.prepare(request)

            # Sniff SSE stream for usage data while forwarding.
            # Scans for message_start (input tokens, model, id) and
            # message_delta (output tokens, stop reason).
            sse_buffer = ""

            async for chunk in response.aiter_bytes():
                await resp.write(chunk)

                sse_buffer += chunk.decode("utf-8", errors="replace")

                # Process complete lines, keep partial tail
                while "\n" in sse_buffer:
                    line, sse_buffer = sse_buffer.split("\n", 1)
                    line = line.strip()
                    if line.startswith("data: "):
                        raw = line[6:]
                        # Only parse events we care about
                        if '"message_start"' in raw or '"message_delta"' in raw:
                            self._process_sse_data(raw)

            await resp.write_eof()
            return resp

    # -- SSE usage extraction -------------------------------------------------

    def _process_sse_data(self, data: str) -> None:
        """Process an SSE data payload for usage information.

        Handles two event types from the Anthropic streaming API:
        - message_start: input tokens (with cache breakdown), model, response ID
        - message_delta: output tokens, stop reason
        """
        try:
            payload = json.loads(data)
        except (json.JSONDecodeError, ValueError):
            return

        event_type = payload.get("type", "")

        if event_type == "message_start":
            message = payload.get("message", {})
            usage = message.get("usage", {})

            self._input_tokens = usage.get("input_tokens", 0)
            self._cache_creation_tokens = usage.get("cache_creation_input_tokens", 0)
            self._cache_read_tokens = usage.get("cache_read_input_tokens", 0)
            # Total for context window tracking (all three count toward the window)
            self._token_count = (
                self._input_tokens
                + self._cache_creation_tokens
                + self._cache_read_tokens
            )
            self._response_model = message.get("model")
            self._response_id = message.get("id")

        elif event_type == "message_delta":
            usage = payload.get("usage", {})
            delta = payload.get("delta", {})

            self._output_tokens = usage.get("output_tokens", 0)
            self._stop_reason = delta.get("stop_reason")

    # -- Usage headers --------------------------------------------------------

    def _sniff_usage_headers(self, headers: httpx.Headers) -> None:
        """Extract usage quota from Anthropic response headers."""
        util_7d = headers.get("anthropic-ratelimit-unified-7d-utilization")
        util_5h = headers.get("anthropic-ratelimit-unified-5h-utilization")

        # No debug logging here — the raw headers have been inspected and
        # confirmed: Anthropic sends 1-2 decimal places of precision.
        # The float() conversion below is lossless.

        if util_7d is not None:
            try:
                self._usage_7d = float(util_7d)
            except ValueError:
                pass

        if util_5h is not None:
            try:
                self._usage_5h = float(util_5h)
            except ValueError:
                pass

    # -- Debug capture --------------------------------------------------------

    def _capture_request(self, path: str, body: dict, suffix: str = "") -> None:
        """Dump request to JSON file for debugging."""
        try:
            CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path_safe = path.replace("/", "_").strip("_")
            suffix_part = f"_{suffix}" if suffix else ""
            filename = f"{timestamp}_{path_safe}{suffix_part}.json"
            with open(CAPTURE_DIR / filename, "w") as f:
                json.dump(body, f, indent=2, default=str)
        except Exception:
            pass
