"""claude.py — The Claude class. One subprocess, four I/O channels.

The only stateful object in the SDK. Wraps the claude binary over
newline-delimited JSON stdio, with an HTTP proxy for compact rewriting
and token counting.

Usage:
    claude = Claude(system_prompt="You are a frog.")
    await claude.start()            # New session
    await claude.start("abc-123")   # Resume session
    await claude.send([{"type": "text", "text": "Hello!"}])
    async for event in claude.events():
        if isinstance(event, AssistantEvent):
            print(event.text)
        elif isinstance(event, ResultEvent):
            break
    await claude.stop()
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import AsyncIterator

from .proxy import CompactConfig, _Proxy


def _bundled_claude_path() -> str:
    """Resolve the claude binary bundled inside claude-agent-sdk."""
    try:
        import claude_agent_sdk._bundled as _bundled
    except ImportError:
        raise RuntimeError(
            "claude-agent-sdk is not installed. "
            "alpha_sdk requires claude-agent-sdk — install it or "
            "check that your environment has the right dependencies."
        )

    bundled_dir = Path(_bundled.__path__[0])
    binary = bundled_dir / "claude"

    if not binary.exists():
        raise RuntimeError(
            f"Bundled claude binary not found at {binary}. "
            f"claude-agent-sdk may be corrupt — reinstall it."
        )

    return str(binary)


# -- State machine ------------------------------------------------------------


class ClaudeState(Enum):
    """Lifecycle states for the claude subprocess."""

    IDLE = auto()      # Not started
    STARTING = auto()  # Subprocess spawned, init handshake in progress
    READY = auto()     # Init complete, accepting messages
    STOPPED = auto()   # Shut down (graceful or error)


# -- Events -------------------------------------------------------------------


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
class UserEvent(Event):
    """User message — content blocks. Only emitted during replay."""

    content: list = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(
            block.get("text", "")
            for block in self.content
            if block.get("type") == "text"
        )


@dataclass
class AssistantEvent(Event):
    """Assistant response — content blocks."""

    content: list = field(default_factory=list)

    @property
    def text(self) -> str:
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
    """Error from claude (process death, parse failure, etc.)."""

    message: str = ""


@dataclass
class StreamEvent(Event):
    """Streaming delta — Messages API format wrapped by claude.

    Arrives BEFORE the complete AssistantEvent for the same content.
    """

    inner: dict = field(default_factory=dict)

    @property
    def event_type(self) -> str:
        return self.inner.get("type", "")

    @property
    def index(self) -> int:
        return self.inner.get("index", 0)

    @property
    def delta_type(self) -> str:
        return self.inner.get("delta", {}).get("type", "")

    @property
    def delta_text(self) -> str:
        delta = self.inner.get("delta", {})
        return delta.get("text", "") or delta.get("thinking", "")

    @property
    def block_type(self) -> str:
        return self.inner.get("content_block", {}).get("type", "")


# Internal — not yielded to consumers
@dataclass
class _ControlRequestEvent(Event):
    """Permission request from claude. Handled internally by auto-approve."""

    request_id: str = ""
    tool_name: str = ""
    request: dict = field(default_factory=dict)


# -- Claude -------------------------------------------------------------------


class Claude:
    """A claude subprocess. The only stateful object in the SDK.

    Manages four I/O channels:
    - stdin/stdout: JSON message framing
    - stderr: background drain
    - HTTP: localhost proxy for compact rewriting + token counting

    Simple API: start(), send(), events(), stop().
    No queues, no routers, no sessions — just the subprocess.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        system_prompt: str | None = None,
        mcp_config: str | None = None,
        permission_mode: str = "bypassPermissions",
        compact_config: CompactConfig | None = None,
        extra_args: list[str] | None = None,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.mcp_config = mcp_config
        self.permission_mode = permission_mode
        self.compact_config = compact_config
        self.extra_args: list[str] = extra_args or []

        self._state = ClaudeState.IDLE
        self._proc: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task | None = None
        self._proxy: _Proxy | None = None
        self._session_id: str | None = None

    @property
    def state(self) -> ClaudeState:
        return self._state

    @property
    def session_id(self) -> str | None:
        """Session ID discovered during init or provided at start."""
        return self._session_id

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None

    # -- Proxy delegates ------------------------------------------------------

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
        return self._proxy.usage_7d if self._proxy else None

    @property
    def usage_5h(self) -> float | None:
        return self._proxy.usage_5h if self._proxy else None

    def reset_token_count(self) -> None:
        """Reset token count to 0. Call after compaction."""
        if self._proxy:
            self._proxy.reset_token_count()

    # -- Lifecycle ------------------------------------------------------------

    async def start(self, session_id: str | None = None) -> None:
        """Spawn claude and perform the init handshake.

        Args:
            session_id: Resume this session, or None for a new session.
        """
        if self._state != ClaudeState.IDLE:
            raise RuntimeError(f"Cannot start in state {self._state}")

        self._state = ClaudeState.STARTING
        self._session_id = session_id

        try:
            upstream_url = os.environ.get("ANTHROPIC_BASE_URL")
            self._proxy = _Proxy(
                compact_config=self.compact_config,
                upstream_url=upstream_url,
            )
            await self._proxy.start()

            self._proc = await self._spawn()
            self._stderr_task = asyncio.create_task(self._drain_stderr())
            await self._init_handshake()
            self._state = ClaudeState.READY
        except Exception:
            self._state = ClaudeState.STOPPED
            await self._cleanup()
            raise

    async def send(self, content: list[dict]) -> None:
        """Send a user message to claude.

        Args:
            content: Messages API content blocks.
        """
        if self._state != ClaudeState.READY:
            raise RuntimeError(f"Cannot send in state {self._state}")

        await self._send_json(self._format_user_message(content))

    async def events(self) -> AsyncIterator[Event]:
        """Yield events from claude's response stream.

        Reads until a ResultEvent (end of turn). Auto-approves
        permission requests — consumers never see them.
        """
        if self._state != ClaudeState.READY:
            raise RuntimeError(f"Cannot read events in state {self._state}")

        while True:
            raw = await self._read_json()

            if raw is None:
                self._state = ClaudeState.STOPPED
                yield ErrorEvent(raw={}, message="claude process exited unexpectedly")
                return

            event = self._parse_event(raw)

            if isinstance(event, _ControlRequestEvent):
                await self._send_permission_response(event.request_id)
                continue

            if isinstance(event, ResultEvent) and not self._session_id:
                if event.session_id:
                    self._session_id = event.session_id

            yield event

            if isinstance(event, ResultEvent):
                return

    async def stop(self) -> None:
        """Gracefully shut down the claude process."""
        if self._state == ClaudeState.STOPPED:
            return

        self._state = ClaudeState.STOPPED
        await self._cleanup()

    # -- Protocol helpers (static, unit-testable) -----------------------------

    @staticmethod
    def _format_user_message(content: list[dict]) -> dict:
        return {
            "type": "user",
            "session_id": "",
            "message": {"role": "user", "content": content},
            "parent_tool_use_id": None,
        }

    @staticmethod
    def _format_init_request() -> dict:
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
        """Parse a raw JSON dict into a typed Event."""
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
            resp = raw.get("response", {})
            return InitEvent(
                raw=raw,
                model=resp.get("model", ""),
                tools=resp.get("tools", []),
                mcp_servers=resp.get("mcpServers", []),
            )

        elif msg_type == "stream_event":
            inner = raw.get("event", {})
            return StreamEvent(raw=raw, inner=inner)

        else:
            return Event(raw=raw)

    # -- Subprocess management ------------------------------------------------

    async def _spawn(self) -> asyncio.subprocess.Process:
        """Spawn the claude binary with stream-json protocol."""
        cmd = [
            _bundled_claude_path(),
            "--output-format", "stream-json",
            "--input-format", "stream-json",
            "--verbose",
            "--model", self.model,
            "--permission-mode", self.permission_mode,
            "--include-partial-messages",
        ]

        if self.system_prompt is not None:
            cmd.extend(["--system-prompt", self.system_prompt])

        if self.mcp_config:
            cmd.extend(["--mcp-config", self.mcp_config])

        if self._session_id:
            cmd.extend(["--resume", self._session_id])

        if self.extra_args:
            cmd.extend(self.extra_args)

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

    async def _send_json(self, obj: dict) -> None:
        """Send a JSON object to claude's stdin."""
        assert self._proc and self._proc.stdin
        self._proc.stdin.write((json.dumps(obj) + "\n").encode())
        await self._proc.stdin.drain()

    async def _read_json(self) -> dict | None:
        """Read one JSON object from claude's stdout."""
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
        await self._send_json(self._format_permission_response(request_id))

    async def _drain_stderr(self) -> None:
        """Read stderr in the background so it doesn't block."""
        assert self._proc and self._proc.stderr
        while True:
            line = await self._proc.stderr.readline()
            if not line:
                break

    async def _cleanup(self) -> None:
        """Clean up subprocess, background tasks, and proxy."""
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
