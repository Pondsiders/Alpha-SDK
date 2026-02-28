"""engine.py — The claude subprocess. All four I/O channels.

Based on the protocol observed in quack-raw.py and quack-wire.py
(Feb 26, 2026). The claude binary communicates via newline-delimited
JSON on stdin/stdout, and makes API requests via ANTHROPIC_BASE_URL.

The Engine manages all four channels:
  1. stdin  — JSON messages in
  2. stdout — JSON events out
  3. stderr — diagnostic output (drained in background)
  4. HTTP   — API requests (intercepted by proxy for compact rewriting)

Usage:
    engine = Engine(model="claude-sonnet-4-20250514")
    await engine.start()            # New session
    await engine.start("abc-123")   # Resume session
    # engine.session_id is None until first turn completes
    await engine.send("Hello!")
    async for event in engine.events():
        if isinstance(event, AssistantEvent):
            print(event.text)
        elif isinstance(event, ResultEvent):
            break
    await engine.stop()
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import AsyncIterator

from .proxy import CompactConfig, _Proxy


# -- State machine ------------------------------------------------------------


class EngineState(Enum):
    """Lifecycle states for the claude subprocess."""

    IDLE = auto()      # Not started
    STARTING = auto()  # Subprocess spawned, init handshake in progress
    READY = auto()     # Init complete, accepting messages
    STOPPED = auto()   # Shut down (graceful or error)


# -- Events -------------------------------------------------------------------
# Parsed JSON messages from claude's stdout.
# The engine yields these; the router dispatches them.


@dataclass
class Event:
    """Base event from the claude process."""

    raw: dict
    is_replay: bool = False


@dataclass
class InitEvent(Event):
    """Capabilities advertisement from the init handshake."""

    model: str = ""
    tools: list = field(default_factory=list)
    mcp_servers: list = field(default_factory=list)


@dataclass
class AssistantEvent(Event):
    """Assistant response — contains content blocks.

    Content blocks follow the Messages API format:
    [{"type": "text", "text": "..."}, {"type": "tool_use", ...}, ...]
    """

    content: list = field(default_factory=list)

    @property
    def text(self) -> str:
        """Extract concatenated text from content blocks."""
        return "".join(
            block.get("text", "")
            for block in self.content
            if block.get("type") == "text"
        )


@dataclass
class ResultEvent(Event):
    """End-of-turn result with session info and cost."""

    session_id: str = ""
    cost_usd: float = 0.0
    num_turns: int = 0
    duration_ms: int = 0
    is_error: bool = False


@dataclass
class SystemEvent(Event):
    """System message from claude (session_id, etc.)."""

    subtype: str = ""


@dataclass
class ErrorEvent(Event):
    """Error from the engine (process death, parse failure, etc.)."""

    message: str = ""


# Internal — not yielded to consumers
@dataclass
class _ControlRequestEvent(Event):
    """Permission request from claude. Handled internally by auto-approve."""

    request_id: str = ""
    tool_name: str = ""
    request: dict = field(default_factory=dict)


# -- Engine -------------------------------------------------------------------


class Engine:
    """Manages a claude subprocess over JSON stdio + HTTP proxy.

    The engine handles all four I/O channels:
    - stdin/stdout: JSON message framing
    - stderr: background drain
    - HTTP: localhost proxy for compact rewriting + token counting

    The engine does NOT interpret events — that's the router's job.
    It yields parsed Event objects and lets consumers decide what to do.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        system_prompt: str | None = None,
        mcp_config: str | None = None,
        permission_mode: str = "bypassPermissions",
        extra_args: list[str] | None = None,
        compact_config: CompactConfig | None = None,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.mcp_config = mcp_config
        self.permission_mode = permission_mode
        self.extra_args = extra_args or []
        self.compact_config = compact_config

        self._state = EngineState.IDLE
        self._proc: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task | None = None
        self._proxy: _Proxy | None = None
        self._session_id: str | None = None

    @property
    def state(self) -> EngineState:
        return self._state

    @property
    def session_id(self) -> str | None:
        """Session ID discovered during init or provided at start."""
        return self._session_id

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None

    # -- Proxy properties (delegates to _proxy) -------------------------------

    @property
    def token_count(self) -> int:
        """Current token count from the HTTP proxy. 0 if no proxy."""
        return self._proxy.token_count if self._proxy else 0

    @property
    def context_window(self) -> int:
        """Context window size. Default if no proxy."""
        return self._proxy.context_window if self._proxy else _Proxy.DEFAULT_CONTEXT_WINDOW

    @property
    def usage_7d(self) -> float | None:
        """7-day usage quota (0.0-1.0) from Anthropic response headers."""
        return self._proxy.usage_7d if self._proxy else None

    @property
    def usage_5h(self) -> float | None:
        """5-hour usage quota (0.0-1.0) from Anthropic response headers."""
        return self._proxy.usage_5h if self._proxy else None

    def reset_token_count(self) -> None:
        """Reset token count to 0. Call after compaction."""
        if self._proxy:
            self._proxy.reset_token_count()

    # -- Lifecycle ------------------------------------------------------------

    async def start(self, session_id: str | None = None) -> None:
        """Spawn claude and perform the init handshake.

        Args:
            session_id: If provided, resume this session (adds --resume flag).
                        If None, start a new session.

        The session ID is not known until the first turn completes.
        After the first turn, read it from engine.session_id.
        """
        if self._state != EngineState.IDLE:
            raise RuntimeError(f"Cannot start engine in state {self._state}")

        self._state = EngineState.STARTING
        self._session_id = session_id  # Known for resume, None for new

        try:
            # Start the HTTP proxy BEFORE spawning claude so
            # ANTHROPIC_BASE_URL is set when the subprocess starts.
            self._proxy = _Proxy(compact_config=self.compact_config)
            await self._proxy.start()

            self._proc = await self._spawn()
            self._stderr_task = asyncio.create_task(self._drain_stderr())
            await self._init_handshake()
            self._state = EngineState.READY
        except Exception:
            self._state = EngineState.STOPPED
            await self._cleanup()
            raise

    async def send(self, text: str) -> None:
        """Send a user message to claude.

        The message is queued on stdin. Call events() to read the response.
        """
        if self._state != EngineState.READY:
            raise RuntimeError(f"Cannot send in state {self._state}")

        await self._send_json(self._format_user_message(text))

    async def events(self) -> AsyncIterator[Event]:
        """Yield events from claude's response stream.

        Reads until a ResultEvent (end of turn). Automatically handles
        permission requests by approving them — consumers never see
        control_request events.

        Yields: AssistantEvent, SystemEvent, ResultEvent, ErrorEvent.
        """
        if self._state != EngineState.READY:
            raise RuntimeError(f"Cannot read events in state {self._state}")

        while True:
            raw = await self._read_json()

            if raw is None:
                self._state = EngineState.STOPPED
                yield ErrorEvent(raw={}, message="claude process exited unexpectedly")
                return

            event = self._parse_event(raw)

            if isinstance(event, _ControlRequestEvent):
                # Auto-approve and continue — don't yield to consumer
                await self._send_permission_response(event.request_id)
                continue

            # Capture session_id from first ResultEvent if not yet known
            if isinstance(event, ResultEvent) and not self._session_id:
                if event.session_id:
                    self._session_id = event.session_id

            yield event

            if isinstance(event, ResultEvent):
                return  # End of turn

    async def stop(self) -> None:
        """Gracefully shut down the claude process."""
        if self._state == EngineState.STOPPED:
            return

        self._state = EngineState.STOPPED
        await self._cleanup()

    # -- Protocol helpers (static, unit-testable) -----------------------------

    @staticmethod
    def _format_user_message(text: str) -> dict:
        """Format a user message for claude's stdin."""
        return {
            "type": "user",
            "session_id": "",
            "message": {"role": "user", "content": text},
            "parent_tool_use_id": None,
        }

    @staticmethod
    def _format_init_request() -> dict:
        """Format the init handshake request."""
        return {
            "type": "control_request",
            "request_id": f"req_0_{os.urandom(4).hex()}",
            "request": {
                "subtype": "initialize",
                "hooks": {},
                "agents": {},
            },
        }

    @staticmethod
    def _format_permission_response(request_id: str) -> dict:
        """Format an auto-approve response to a permission request."""
        return {
            "type": "control_response",
            "response": {
                "subtype": "success",
                "request_id": request_id,
                "response": {"approved": True},
            },
        }

    @staticmethod
    def _parse_event(raw: dict) -> Event:
        """Parse a raw JSON dict into a typed Event.

        This is the single point where wire protocol maps to our type system.
        """
        msg_type = raw.get("type", "")

        if msg_type == "assistant":
            content = raw.get("message", {}).get("content", [])
            return AssistantEvent(raw=raw, content=content)

        elif msg_type == "result":
            return ResultEvent(
                raw=raw,
                session_id=raw.get("session_id", ""),
                cost_usd=raw.get("total_cost_usd", 0.0),
                num_turns=raw.get("num_turns", 0),
                duration_ms=raw.get("duration_ms", 0),
                is_error=raw.get("is_error", False),
            )

        elif msg_type == "system":
            return SystemEvent(raw=raw, subtype=raw.get("subtype", ""))

        elif msg_type == "control_request":
            req = raw.get("request", {})
            return _ControlRequestEvent(
                raw=raw,
                request_id=raw.get("request_id", ""),
                tool_name=req.get("tool_name", req.get("subtype", "")),
                request=req,
            )

        elif msg_type == "control_response":
            # Init response — extract capabilities
            resp = raw.get("response", {})
            return InitEvent(
                raw=raw,
                model=resp.get("model", ""),
                tools=resp.get("tools", []),
                mcp_servers=resp.get("mcpServers", []),
            )

        else:
            return Event(raw=raw)

    # -- Subprocess management ------------------------------------------------

    async def _spawn(self) -> asyncio.subprocess.Process:
        """Spawn the claude binary with stream-json protocol."""
        cmd = [
            "claude",
            "--output-format", "stream-json",
            "--input-format", "stream-json",
            "--verbose",
            "--model", self.model,
            "--permission-mode", self.permission_mode,
        ]

        if self.system_prompt:
            cmd.extend(["--system-prompt", self.system_prompt])

        if self.mcp_config:
            cmd.extend(["--mcp-config", self.mcp_config])

        if self._session_id:
            cmd.extend(["--resume", self._session_id])

        if self.extra_args:
            cmd.extend(self.extra_args)

        # Clean env for subprocess:
        # - Clear CLAUDECODE so we can spawn from within another claude process
        # - Set ANTHROPIC_BASE_URL to route API requests through our proxy
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        if self._proxy and self._proxy.port:
            env["ANTHROPIC_BASE_URL"] = self._proxy.base_url

        return await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

    async def _init_handshake(self) -> InitEvent:
        """Perform the init handshake and return capabilities."""
        await self._send_json(self._format_init_request())

        while True:
            raw = await self._read_json()
            if raw is None:
                raise RuntimeError("claude exited during init handshake")

            event = self._parse_event(raw)
            if isinstance(event, InitEvent):
                return event
            # Swallow system messages etc. during init

    async def _send_json(self, obj: dict) -> None:
        """Send a JSON object to claude's stdin."""
        assert self._proc and self._proc.stdin
        self._proc.stdin.write((json.dumps(obj) + "\n").encode())
        await self._proc.stdin.drain()

    async def _read_json(self) -> dict | None:
        """Read one JSON object from claude's stdout.

        Returns None if the process has exited (EOF on stdout).
        Skips blank lines and malformed JSON silently.
        """
        assert self._proc and self._proc.stdout
        while True:
            line = await self._proc.stdout.readline()
            if not line:
                return None
            text = line.decode().strip()
            if not text:
                continue
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                continue

    async def _send_permission_response(self, request_id: str) -> None:
        """Auto-approve a permission request."""
        await self._send_json(self._format_permission_response(request_id))

    async def _drain_stderr(self) -> None:
        """Read stderr in the background so it doesn't block."""
        assert self._proc and self._proc.stderr
        while True:
            line = await self._proc.stderr.readline()
            if not line:
                break

    async def _cleanup(self) -> None:
        """Clean up the subprocess, background tasks, and proxy."""
        if self._proc:
            if self._proc.stdin and not self._proc.stdin.is_closing():
                self._proc.stdin.close()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._proc.kill()
                await self._proc.wait()

        if self._stderr_task and not self._stderr_task.done():
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass

        if self._proxy:
            await self._proxy.stop()
            self._proxy = None
