"""proxy.py — HTTP proxy for the claude subprocess's API channel.

The Engine manages four I/O channels:
  1. stdin  — JSON messages in
  2. stdout — JSON events out
  3. stderr — diagnostic output (drained in background)
  4. HTTP   — API requests via ANTHROPIC_BASE_URL

This module handles #4. It is a private implementation detail of Engine.
No other module should import this directly.

The proxy sits between claude and Anthropic's API. It:
- Rewrites compact requests (three-phase surgical intervention)
- Counts tokens (fire-and-forget echo to /v1/messages/count_tokens)
- Sniffs usage quota headers from responses

The compact rewriting logic is ported from v1.x compact_proxy.py.
Detection signatures are constants (what claude sends).
Replacement content comes from CompactConfig (what we inject).
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx
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
    ):
        self._compact_config = compact_config
        self._context_window = context_window

        self._port: int | None = None
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._http_client: httpx.AsyncClient | None = None

        # Token counting state
        self._token_count = 0
        self._warned_no_api_key = False

        # Usage quota state (from Anthropic response headers)
        self._usage_7d: float | None = None
        self._usage_5h: float | None = None

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

    def reset_token_count(self) -> None:
        """Reset token count to 0. Call after compaction."""
        self._token_count = 0

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
        """Route incoming requests."""
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
        if body is not None and self._compact_config is not None:
            rewrite_compact(body, self._compact_config)
            body_bytes = json.dumps(body).encode()

        # Debug capture — after rewrite
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

        # Token counting: fire-and-forget on /v1/messages requests
        if body is not None and path == "/v1/messages":
            asyncio.create_task(self._count_tokens(body, headers))

        # Forward to Anthropic
        url = f"{ANTHROPIC_API_URL}{path}"

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
            async for chunk in response.aiter_bytes():
                await resp.write(chunk)
            await resp.write_eof()
            return resp

    # -- Token counting -------------------------------------------------------

    async def _count_tokens(self, body: dict, headers: dict) -> None:
        """Count tokens via fire-and-forget to /v1/messages/count_tokens.

        Uses ALPHA_ANTHROPIC_API_KEY (not ANTHROPIC_API_KEY) to avoid
        interference with Claude Max auth.
        """
        if self._http_client is None:
            return

        api_key = os.environ.get("ALPHA_ANTHROPIC_API_KEY")
        if not api_key:
            if not self._warned_no_api_key:
                self._warned_no_api_key = True
            return

        try:
            count_headers = {
                "x-api-key": api_key,
                "anthropic-version": headers.get("anthropic-version", "2023-06-01"),
                "content-type": "application/json",
            }

            # Only fields the count endpoint accepts
            count_body = {}
            for key in ("messages", "model", "system", "tools", "tool_choice", "thinking"):
                if key in body:
                    count_body[key] = body[key]

            # Strip cache_control from tools (count endpoint is stricter)
            if "tools" in count_body:
                cleaned = []
                for tool in count_body["tools"]:
                    tool = dict(tool)
                    tool.pop("cache_control", None)
                    if "custom" in tool and isinstance(tool["custom"], dict):
                        tool["custom"] = {
                            k: v for k, v in tool["custom"].items()
                            if k != "cache_control"
                        }
                    cleaned.append(tool)
                count_body["tools"] = cleaned

            response = await self._http_client.post(
                f"{ANTHROPIC_API_URL}/v1/messages/count_tokens",
                content=json.dumps(count_body).encode(),
                headers=count_headers,
                timeout=10.0,
            )

            if response.status_code != 200:
                return

            result = response.json()
            new_count = result.get("input_tokens", 0)

            # High-water mark — only update upward
            if new_count > self._token_count:
                self._token_count = new_count

        except Exception:
            pass  # Fire and forget

    # -- Usage headers --------------------------------------------------------

    def _sniff_usage_headers(self, headers: httpx.Headers) -> None:
        """Extract usage quota from Anthropic response headers."""
        util_7d = headers.get("anthropic-ratelimit-unified-7d-utilization")
        util_5h = headers.get("anthropic-ratelimit-unified-5h-utilization")

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
