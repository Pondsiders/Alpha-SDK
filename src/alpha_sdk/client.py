"""AlphaClient - the main interface to Alpha.

Architecture:
- System prompt is assembled once at connect() and passed to SDK
- Orientation (capsules, letter, here, context, etc.) goes in first user message
- Memories and memorables go in user content, not system prompt
- Minimal proxy intercepts only compact prompts for rewriting
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, AsyncGenerator, Literal

import logfire
import pendulum

# Permission modes supported by Claude Agent SDK
PermissionMode = Literal[
    "default",           # Standard permission behavior
    "acceptEdits",       # Auto-accept file edits
    "plan",              # Planning mode - no execution
    "bypassPermissions"  # Bypass all permission checks (use with caution)
]

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    UserMessage,
)
from claude_agent_sdk.types import (
    StreamEvent,
    HookMatcher,
    PreCompactHookInput,
    HookContext,
    ToolUseBlock,
    ToolResultBlock,
)

from .archive import archive_turn
from .broadcast import SessionBroadcast
from .compact_proxy import CompactProxy, TokenCountCallback
from .memories.db import search_memories
from .memories.embeddings import embed_query, EmbeddingError
from .memories.images import load_thumbnail_base64, process_inline_image
from .memories.recall import recall
from .memories.suggest import suggest
from .memories.vision import caption_image
from .preprocessing import (
    build_orientation,
    extract_prompt_text,
    format_image_recall,
    format_memory,
    get_approach_light,
    relative_time,
)
from .sessions import list_sessions, get_session_path, get_sessions_dir, SessionInfo
from .tools.cortex import create_cortex_server
from .tools.fetch import create_fetch_server
from .tools.forge import create_forge_server
from .tools.handoff import create_handoff_server
from .system_prompt.soul import get_soul

# The Alpha Plugin — agents, skills, and tools bundled in a sibling repo
_ALPHA_PLUGIN_DIR = str(Path(__file__).parent.parent.parent.parent / "alpha_plugin")

# Store original ANTHROPIC_BASE_URL so we can restore it
_ORIGINAL_ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL")

# Tools that are Claude Code concepts and break non-CC clients.
# These are always disallowed, regardless of what the consumer passes.
_SDK_DISALLOWED_TOOLS = [
    "EnterPlanMode",
    "ExitPlanMode",
    "AskUserQuestion",
]

# Sentinel for generator shutdown in streaming input mode
_SHUTDOWN = object()


class AlphaClient:
    """Long-lived client that wraps Claude Agent SDK.

    Proxyless architecture:
    - System prompt = just the soul (small, truly static)
    - Orientation = injected in first user message of session
    - Memories = per-turn, in user content
    - Memorables = per-turn nudge, in user content

    Usage:
        client = AlphaClient(cwd="/Pondside", client_name="duckpond")
        await client.connect(session_id=None, streaming=True)

        await client.send("Hello!")
        async for event in client.events():
            print(event)  # {"type": "text-delta", "data": {"text": "Hi"}, "id": 1}
    """

    # The model that IS Alpha. Pinned at the SDK level, not configurable per-client.
    # When we upgrade Alpha to a new model, we bump alpha_sdk version.
    ALPHA_MODEL = "claude-opus-4-6"

    def __init__(
        self,
        cwd: str = "/Pondside",
        client_name: str = "alpha_sdk",
        hostname: str | None = None,
        disallowed_tools: list[str] | None = None,
        mcp_servers: dict | None = None,
        archive: bool = True,
        include_partial_messages: bool = True,
        permission_mode: PermissionMode = "default",
        on_token_count: "TokenCountCallback | None" = None,
    ):
        """Initialize the Alpha client.

        Args:
            cwd: Working directory for the agent
            client_name: Name of the client (for logging, HUD)
            hostname: Machine hostname (auto-detected if not provided)
            disallowed_tools: List of tool names to block (e.g., EnterPlanMode)
            mcp_servers: Dict of MCP server configurations
            archive: Whether to archive turns to Postgres
            include_partial_messages: Stream partial messages for real-time updates
            permission_mode: How to handle tool permission requests
            on_token_count: Callback when token count increases: (count, window) -> None
                           Used for context-o-meter. Fires with max(seen_counts) after each request.
        """
        self.cwd = cwd
        self.client_name = client_name
        self.hostname = hostname
        self.disallowed_tools = disallowed_tools
        self.mcp_servers = mcp_servers or {}
        self.archive = archive
        self.include_partial_messages = include_partial_messages
        self.permission_mode = permission_mode
        self._on_token_count = on_token_count

        # Internal state
        self._sdk_client: ClaudeSDKClient | None = None
        self._current_session_id: str | None = None
        self._system_prompt: str | None = None  # Just the soul, assembled once
        self._orientation_blocks: list[dict] | None = None  # Cached for re-injection
        self._compact_proxy: CompactProxy | None = None  # For compact prompt rewriting

        # Turn state
        self._last_user_content: str = ""  # Just the user's text (for memorables)
        self._last_content_blocks: list[dict] = []  # Full content array (for observability)
        self._last_assistant_content: str = ""
        self._turn_span: logfire.LogfireSpan | None = None
        self._suggest_task: asyncio.Task | None = None
        self._pending_memorables: list[str] = []  # From last suggest, consumed on next send()

        # Compaction flag - set by PreCompact hook, cleared after re-orientation
        self._needs_reorientation: bool = False

        # Hand-off: compact instructions set by the hand-off tool, consumed after turn
        self._pending_compact: str | None = None
        # Hand-off state machine for streaming mode: None | "compacting" | "waking_up"
        self._compact_mode: str | None = None
        self._compact_summary_parts: list[str] = []  # Dead — kept for reference
        self._compact_user_messages: list[str] = []  # UserMessages captured during compact
        self._compact_span: logfire.LogfireSpan | None = None

        # Approach lights: escalating context warnings at 65% and 75%
        # Tracks the highest tier warned so we don't repeat the same warning
        self._approach_warned: int = 0  # 0=none, 1=amber(65%), 2=red(75%)

        # Context-start flag: set when orientation is injected (new session or post-compact)
        # Consumed by the turn span to apply the Logfire "context-start" tag
        self._is_context_start: bool = False

        # Streaming input mode state (when connect(streaming=True))
        self._streaming: bool = False
        self._queue: asyncio.Queue | None = None       # Input: messages for the generator
        self._broadcast: SessionBroadcast | None = None  # Output: fan-out to SSE subscribers
        self._session_task: asyncio.Task | None = None  # Background: _run_session()
        self._turn_count: int = 0                       # Turns sent this session

    # -------------------------------------------------------------------------
    # Session Discovery (static methods)
    # -------------------------------------------------------------------------

    @staticmethod
    def list_sessions(cwd: str = "/Pondside", limit: int = 50) -> list[SessionInfo]:
        """List available sessions for resumption."""
        return list_sessions(cwd, limit)

    @staticmethod
    def get_session_path(session_id: str, cwd: str = "/Pondside") -> str:
        """Get the filesystem path for a session."""
        return str(get_session_path(session_id, cwd))

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def connect(self, session_id: str | None = None, streaming: bool = False) -> None:
        """Connect to Claude.

        Starts the compact proxy, assembles the system prompt (soul only),
        and creates the SDK client. Orientation will be injected on first send().

        Args:
            session_id: Session to resume, or None for new session
            streaming: If True, use streaming input mode (send/events API)
        """
        with logfire.span("connect", client_name=self.client_name, streaming=streaming):
            # Start compact proxy (intercepts compact prompts + counts tokens)
            self._compact_proxy = CompactProxy(on_token_count=self._on_token_count)
            await self._compact_proxy.start()
            os.environ["ANTHROPIC_BASE_URL"] = self._compact_proxy.base_url
            logfire.info("CompactProxy started ({url})", url=self._compact_proxy.base_url)

            # Build the system prompt - just the soul
            self._system_prompt = f"# Alpha\n\n{get_soul()}"
            logfire.info("Soul loaded ({chars} chars)", chars=len(self._system_prompt))

            # Pre-build orientation blocks (will be injected on first turn)
            self._orientation_blocks = await self._build_orientation()
            logfire.info(
                "Orientation assembled ({blocks} blocks)",
                blocks=len(self._orientation_blocks),
            )

            # Create SDK client with system prompt
            await self._create_sdk_client(session_id)
            logfire.info("SDK client created")

        # Streaming setup AFTER span closes — _run_session() must not
        # inherit connect's trace context, or turn spans become children.
        if streaming:
            self._streaming = True
            self._queue = asyncio.Queue()
            self._broadcast = SessionBroadcast()
            self._turn_count = 0
            self._session_task = asyncio.create_task(self._run_session())

            # Wire up real-time token count → SSE context events
            # CompactProxy fires this on every /v1/messages request,
            # so the meter updates as tool calls happen, not just at turn-end.
            async def _streaming_token_count(count: int, window: int) -> None:
                await self._push_event("context", {"count": count, "window": window})

            self.set_token_count_callback(_streaming_token_count)


    async def disconnect(self) -> None:
        """Disconnect and clean up resources."""
        with logfire.span("disconnect", client_name=self.client_name):
            # Shutdown streaming if active
            if self._streaming:
                if self._queue:
                    await self._queue.put(_SHUTDOWN)
                if self._session_task:
                    try:
                        await asyncio.wait_for(self._session_task, timeout=10.0)
                    except asyncio.TimeoutError:
                        self._session_task.cancel()
                    except Exception:
                        pass
                    self._session_task = None
                self._queue = None
                if self._broadcast:
                    self._broadcast.close()
                    self._broadcast = None
                self._streaming = False
                logfire.info("Streaming shutdown")

            # Wait for any pending suggest task
            if self._suggest_task is not None:
                try:
                    await self._suggest_task
                except Exception:
                    pass
                self._suggest_task = None

            # Disconnect SDK client
            if self._sdk_client:
                await self._sdk_client.disconnect()
                self._sdk_client = None
                logfire.info("SDK client disconnected")

            # Stop compact proxy and restore original ANTHROPIC_BASE_URL
            if self._compact_proxy:
                await self._compact_proxy.stop()
                self._compact_proxy = None
                if _ORIGINAL_ANTHROPIC_BASE_URL:
                    os.environ["ANTHROPIC_BASE_URL"] = _ORIGINAL_ANTHROPIC_BASE_URL
                elif "ANTHROPIC_BASE_URL" in os.environ:
                    del os.environ["ANTHROPIC_BASE_URL"]
                logfire.info("CompactProxy stopped")

        self._current_session_id = None

    # -------------------------------------------------------------------------
    # Conversation
    # -------------------------------------------------------------------------

    async def send(
        self,
        prompt: str | list[dict[str, Any]],
    ) -> None:
        """Preprocess and queue a message for streaming input.

        Preprocesses and pushes to the input queue.
        Returns immediately — responses flow through events().

        Requires connect(streaming=True) to have been called first.
        """
        if not self._streaming or not self._queue:
            raise RuntimeError("Streaming not active. Call connect(streaming=True).")

        # Extract text for span naming
        prompt_text_for_span = extract_prompt_text(prompt)
        prompt_preview = prompt_text_for_span[:50].replace("\n", " ").strip()
        if len(prompt_text_for_span) > 50:
            prompt_preview += "…"

        # Open the turn span HERE so preprocessing breadcrumbs nest under it.
        # The span stays open across the async boundary — _run_session() will
        # finalize and close it when ResultMessage arrives.
        span_kwargs: dict[str, Any] = {
            "prompt_preview": prompt_preview,
            "session_id": self._current_session_id or "new",
            "client_name": self.client_name,
        }

        self._turn_span = logfire.span(
            "turn: {prompt_preview}",
            **span_kwargs,
        )
        self._turn_span.__enter__()

        # Set initial gen_ai attributes (will be progressively enhanced)
        self._turn_span.set_attribute("gen_ai.system", "anthropic")
        self._turn_span.set_attribute("gen_ai.operation.name", "chat")
        self._turn_span.set_attribute("gen_ai.request.model", self.ALPHA_MODEL)
        if self._current_session_id:
            self._turn_span.set_attribute("gen_ai.conversation.id", self._current_session_id)
        if self._system_prompt:
            self._turn_span.set_attribute(
                "gen_ai.system_instructions",
                json.dumps([{"type": "text", "content": self._system_prompt}])
            )

        # Set trace context on proxy so its spans nest under this turn
        if self._compact_proxy:
            self._compact_proxy.set_trace_context(logfire.get_context())

        # Preprocessing pipeline — breadcrumbs will nest under the turn span
        content_blocks, prompt_text = await self._build_user_content(prompt)

        self._last_user_content = prompt_text
        self._last_content_blocks = content_blocks
        self._turn_count += 1

        # Update turn span with structured input
        user_parts = []
        for block in content_blocks:
            if block.get("type") == "text":
                user_parts.append({"type": "text", "content": block["text"]})
        self._turn_span.set_attribute(
            "gen_ai.input.messages",
            json.dumps([{"role": "user", "parts": user_parts}])
        )
        self._turn_span.set_attribute("gen_ai.output.messages", json.dumps([]))

        # Tag context-start for Logfire filtering
        if self._is_context_start:
            self._turn_span.set_attribute("_tags", json.dumps(["context-start"]))

        # Build the message in SDK streaming input format
        message = {
            "type": "user",
            "message": {"role": "user", "content": content_blocks},
            "session_id": self._current_session_id or "new",
        }

        # Push turn-start event BEFORE queueing
        # (so consumers see the activity indicator immediately)
        await self._push_event("turn-start", {})

        await self._queue.put(message)

    async def events(self, from_seq: int = 0) -> AsyncGenerator[dict[str, Any], None]:
        """Async iterator over SSE events.

        Each call creates an independent subscriber — multiple callers
        each receive every event (fan-out, not split).

        Args:
            from_seq: Replay buffered events after this sequence number.
                Pass Last-Event-ID from SSE spec for reconnection.

        Yields dicts like:
            {"type": "text-delta", "data": {"text": "Hello"}, "id": 1}
            {"type": "turn-end", "data": {"sessionId": "abc"}, "id": 42}

        Terminates when the session ends (disconnect or error).
        """
        if not self._broadcast:
            raise RuntimeError("Streaming not active.")
        queue = self._broadcast.subscribe(from_seq)
        try:
            while True:
                event = await queue.get()
                if event is None:
                    return
                yield event
        finally:
            self._broadcast.unsubscribe(queue)

    async def interrupt(self) -> None:
        """Interrupt the current SDK operation."""
        if self._sdk_client:
            await self._sdk_client.interrupt()

    async def _build_user_content(
        self,
        prompt: str | list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], str]:
        """Build content blocks for a user message.

        The preprocessing pipeline. Handles:
        - Orientation injection (first turn / post-compact)
        - Memorables nudge (Intro speaks)
        - Memory recall + Mind's Eye image injection
        - PSO-8601 timestamp
        - Inline image processing (resize, caption, recall)
        - Approach lights (context warnings at 65%, 75%)

        Returns (content_blocks, prompt_text).
        """
        # Extract text for memory operations and logging
        prompt_text = extract_prompt_text(prompt)

        content_blocks: list[dict[str, Any]] = []

        # Orientation: first turn of a NEW session, or post-compact reorientation.
        # NOT on resume — the session history already has orientation from before.
        # _turn_count == 0 means this is the first send() call.
        # _current_session_id is None means this is a genuinely new session (not a resume).
        needs_orientation = self._needs_reorientation or (self._turn_count == 0 and self._current_session_id is None)
        self._is_context_start = needs_orientation
        if needs_orientation:
            if self._needs_reorientation:
                self._orientation_blocks = await self._build_orientation()
                if self._compact_proxy:
                    self._compact_proxy.reset_token_count()
            if self._orientation_blocks:
                content_blocks.extend(self._orientation_blocks)
            self._needs_reorientation = False
            reason = "post-compact" if self._is_context_start and self._turn_count > 0 else "new session"
            logfire.info("Orientation: injected ({reason})", reason=reason)
        else:
            logfire.info("Orientation: skipped (turn {turn})", turn=self._turn_count)

        # Memorables nudge (from previous turn's suggest)
        if self._pending_memorables:
            nudge = "## Intro speaks\n\n"
            nudge += "Alpha, consider storing these from the previous turn:\n"
            nudge += "\n".join(f"- {m}" for m in self._pending_memorables)
            content_blocks.append({"type": "text", "text": nudge})
            logfire.info("Memorables: {count} items injected", count=len(self._pending_memorables))
            self._pending_memorables = []
        else:
            logfire.info("Memorables: none pending")

        # Memory recall
        memories = await recall(prompt_text, self._current_session_id or "new")
        images_loaded = 0
        if memories:
            for mem in memories:
                content_blocks.append({
                    "type": "text",
                    "text": format_memory(mem),
                })
                # Mind's Eye: inject attached images
                if mem.get("image_path"):
                    image_data = load_thumbnail_base64(mem["image_path"])
                    if image_data:
                        content_blocks.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data,
                            },
                        })
                        images_loaded += 1
        logfire.info(
            "Recall: {count} memories{images}",
            count=len(memories) if memories else 0,
            images=f", {images_loaded} images" if images_loaded else "",
            memory_ids=[m["id"] for m in memories] if memories else [],
            memory_scores=[round(m.get("score", 0), 3) for m in memories] if memories else [],
        )

        # PSO-8601 timestamp
        sent_at = pendulum.now("America/Los_Angeles").format("ddd MMM D YYYY, h:mm A")
        content_blocks.append({
            "type": "text",
            "text": f"[Sent {sent_at}]",
        })

        # User prompt (with inline image processing if multimodal)
        if isinstance(prompt, str):
            content_blocks.append({"type": "text", "text": prompt})
        else:
            processed = await self._process_inline_images(prompt)
            content_blocks.extend(processed)
            logfire.info("Inline images: {count} processed", count=len([b for b in processed if b.get("type") == "image"]))

        # Approach lights
        approach_light = self._get_approach_light()
        if approach_light:
            content_blocks.append({"type": "text", "text": approach_light})
            pct = (self._compact_proxy.token_count / self._compact_proxy.context_window * 100) if self._compact_proxy and self._compact_proxy.context_window else 0
            level = "red" if self._approach_warned >= 2 else "amber"
            logfire.info("Approach lights: {level} ({pct:.0f}%)", level=level, pct=pct)
        else:
            pct = (self._compact_proxy.token_count / self._compact_proxy.context_window * 100) if self._compact_proxy and self._compact_proxy.context_window else 0
            logfire.info("Approach lights: clear ({pct:.0f}%)", pct=pct)

        return content_blocks, prompt_text

    async def _message_generator(self) -> AsyncGenerator[dict[str, Any], None]:
        """Async generator backed by the input queue.

        Yields fully-assembled message dicts to the SDK.
        Blocks on queue.get() between turns.
        Returns when SHUTDOWN sentinel is received.
        """
        while True:
            message = await self._queue.get()
            if message is _SHUTDOWN:
                return
            yield message

    async def _run_session(self) -> None:
        """Background task: stream messages to SDK and route responses.

        This is the heart of streaming input mode. It:
        1. Starts the SDK in streaming input mode (query + generator)
        2. Routes all response messages to the response queue as SSE events
        3. Detects turn boundaries (ResultMessage)
        4. Runs post-turn work (suggest, archive)
        """
        try:
            # Start streaming input — query() blocks until generator exhausts,
            # so run it concurrently with receive_messages().
            # SDK docs: "you must call receive_messages() concurrently."
            query_task = asyncio.create_task(
                self._sdk_client.query(
                    self._message_generator(),
                    session_id=self._current_session_id or "new",
                )
            )

            # Process all responses across all turns concurrently with query
            assistant_text_parts: list[str] = []
            turn_span: logfire.LogfireSpan | None = None
            turn_output_parts: list[dict] = []
            inference_count = 0

            async for message in self._sdk_client.receive_messages():
                # ── Compact mode: suppress responses, capture summary ──
                if self._compact_mode == "compacting":
                    if isinstance(message, UserMessage):
                        # The SDK injects the compaction summary as a UserMessage.
                        # Capture it for gen_ai.input.messages on the wake-up turn.
                        content = message.content if isinstance(message.content, str) else ""
                        if content:
                            self._compact_user_messages.append(content)
                    elif isinstance(message, ResultMessage):
                        # Compact complete — build wake-up.
                        # NOTE: The summary is already in the session as a UserMessage
                        # (captured above). We don't need to inject it — the SDK does that.
                        self._orientation_blocks = await self._build_orientation()
                        logfire.info(
                            "Compact: orientation rebuilt ({blocks} blocks)",
                            blocks=len(self._orientation_blocks) if self._orientation_blocks else 0,
                        )
                        wake_up_blocks: list[dict[str, Any]] = []

                        # Orientation blocks
                        if self._orientation_blocks:
                            wake_up_blocks.extend(self._orientation_blocks)

                        # Recall memories relevant to what we were doing
                        wake_up_prompt = (
                            "You've just been through a context compaction. "
                            "Jeffery is here and listening. Orient yourself — "
                            "read the summary above, check in, ask questions "
                            "if anything's unclear."
                        )
                        memories = await recall(wake_up_prompt, self._current_session_id or "new")
                        if memories:
                            for mem in memories:
                                wake_up_blocks.append({
                                    "type": "text",
                                    "text": format_memory(mem),
                                })
                                if mem.get("image_path"):
                                    image_data = load_thumbnail_base64(mem["image_path"])
                                    if image_data:
                                        wake_up_blocks.append({
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": "image/jpeg",
                                                "data": image_data,
                                            },
                                        })

                        logfire.info(
                            "Compact: wake-up recall ({count} memories)",
                            count=len(memories) if memories else 0,
                        )

                        # Timestamp + wake-up prompt
                        sent_at = pendulum.now("America/Los_Angeles").format("ddd MMM D YYYY, h:mm A")
                        wake_up_blocks.append({"type": "text", "text": f"[Sent {sent_at}]"})
                        wake_up_blocks.append({"type": "text", "text": wake_up_prompt})

                        # Queue wake-up message
                        wake_up_message = {
                            "type": "user",
                            "message": {"role": "user", "content": wake_up_blocks},
                            "session_id": self._current_session_id or "new",
                        }

                        # Reset token count (context was just compacted)
                        if self._compact_proxy:
                            self._compact_proxy.reset_token_count()

                        # Reset approach lights (fresh context)
                        self._approach_warned = 0

                        # Mark wake-up as context-start (new context window)
                        self._is_context_start = True
                        # Store wake-up blocks as last content for turn span input
                        self._last_content_blocks = wake_up_blocks
                        # Clear user content — wake-up turns have no user prompt
                        self._last_user_content = ""

                        # Push turn-start for the wake-up turn
                        await self._push_event("turn-start", {})

                        self._compact_mode = "waking_up"
                        await self._queue.put(wake_up_message)

                        # Close compact span — wake-up is queued, compaction is done
                        if self._compact_span:
                            self._compact_span.__exit__(None, None, None)
                            self._compact_span = None

                    else:
                        pass  # SystemMessages during compact — no action needed

                    # Suppress all compact responses from SSE
                    continue

                # ── Pick up turn span from send() on first response ──
                if turn_span is None:
                    turn_span = self._turn_span  # Opened in send()

                    # For wake-up turns (post-compact), send() wasn't called —
                    # _run_session() queued the message directly. Create span here.
                    if turn_span is None:
                        prompt_preview = self._last_user_content[:50].replace("\n", " ").strip()
                        if len(self._last_user_content) > 50:
                            prompt_preview += "…"

                        span_kwargs: dict[str, Any] = {
                            "prompt_preview": prompt_preview,
                            "session_id": self._current_session_id or "new",
                            "client_name": self.client_name,
                        }
                        if self._is_context_start:
                            span_kwargs["_tags"] = ["context-start"]

                        turn_span = logfire.span(
                            "turn: {prompt_preview}",
                            **span_kwargs,
                        )
                        turn_span.__enter__()

                        # gen_ai attributes for Model Run card
                        turn_span.set_attribute("gen_ai.system", "anthropic")
                        turn_span.set_attribute("gen_ai.operation.name", "chat")
                        turn_span.set_attribute("gen_ai.request.model", self.ALPHA_MODEL)
                        if self._current_session_id:
                            turn_span.set_attribute("gen_ai.conversation.id", self._current_session_id)
                        if self._system_prompt:
                            turn_span.set_attribute(
                                "gen_ai.system_instructions",
                                json.dumps([{"type": "text", "content": self._system_prompt}])
                            )
                        # Input messages: SDK's compact UserMessages + our wake-up blocks
                        input_messages = []
                        # SDK messages (compaction summary, callback output, etc.)
                        for um_content in self._compact_user_messages:
                            input_messages.append({
                                "role": "user",
                                "parts": [{"type": "text", "content": um_content}],
                            })
                        self._compact_user_messages = []
                        # Our wake-up orientation + recall + prompt
                        user_parts = []
                        for block in self._last_content_blocks:
                            if block.get("type") == "text":
                                user_parts.append({"type": "text", "content": block["text"]})
                        input_messages.append({"role": "user", "parts": user_parts})
                        turn_span.set_attribute(
                            "gen_ai.input.messages",
                            json.dumps(input_messages),
                        )
                        turn_span.set_attribute("gen_ai.output.messages", json.dumps([]))

                    # Reset per-turn accumulators
                    turn_output_parts = []
                    assistant_text_parts = []
                    inference_count = 0

                # ── StreamEvent: real-time text/thinking deltas ──
                if isinstance(message, StreamEvent):
                    event = message.event
                    event_type = event.get("type")
                    if event_type == "content_block_delta":
                        delta = event.get("delta", {})
                        delta_type = delta.get("type")
                        if delta_type == "text_delta":
                            text = delta.get("text", "")
                            if text:
                                await self._push_event("text-delta", {"text": text})
                        elif delta_type == "thinking_delta":
                            thinking = delta.get("thinking", "")
                            if thinking:
                                await self._push_event("thinking-delta", {"text": thinking})

                # ── AssistantMessage: tool calls (text already streamed) ──
                elif isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            assistant_text_parts.append(block.text)
                            turn_output_parts.append({
                                "type": "text",
                                "content": block.text,
                            })
                        elif isinstance(block, ToolUseBlock):
                            turn_output_parts.append({
                                "type": "tool_call",
                                "id": block.id,
                                "name": block.name,
                                "arguments": block.input,
                            })
                            await self._push_event("tool-call", {
                                "toolCallId": block.id,
                                "toolName": block.name,
                                "args": block.input,
                                "argsText": json.dumps(block.input),
                            })

                    # Update turn span output progressively
                    if turn_span and turn_output_parts:
                        turn_span.set_attribute(
                            "gen_ai.output.messages",
                            json.dumps([{"role": "assistant", "parts": turn_output_parts}])
                        )

                    # Capture finish reason
                    if hasattr(message, 'stop_reason') and message.stop_reason:
                        if turn_span:
                            turn_span.set_attribute(
                                "gen_ai.response.finish_reasons",
                                json.dumps([message.stop_reason])
                            )

                # ── UserMessage: tool results ──
                elif isinstance(message, UserMessage):
                    if isinstance(message.content, list):
                        for block in message.content:
                            if isinstance(block, ToolResultBlock):
                                inference_count += 1
                                result_content = block.content
                                if isinstance(result_content, list):
                                    result_content = json.dumps(result_content)
                                elif result_content is None:
                                    result_content = ""
                                await self._push_event("tool-result", {
                                    "toolCallId": block.tool_use_id,
                                    "result": str(result_content),
                                    "isError": block.is_error or False,
                                })

                # ── ResultMessage: turn complete ──
                elif isinstance(message, ResultMessage):
                    self._current_session_id = message.session_id

                    # ── Finalize turn span and launch post-turn work ──
                    if turn_span:
                        self._last_assistant_content = "".join(assistant_text_parts)
                        turn_span.set_attribute("session_id", message.session_id)
                        turn_span.set_attribute("gen_ai.conversation.id", message.session_id)
                        turn_span.set_attribute("duration_ms", message.duration_ms)
                        turn_span.set_attribute("inference_count", inference_count + 1)
                        turn_span.set_attribute("response_length", len(self._last_assistant_content))
                        if message.total_cost_usd:
                            turn_span.set_attribute("cost_usd", message.total_cost_usd)
                        if message.usage:
                            input_tokens = message.usage.get("input_tokens", 0)
                            output_tokens = message.usage.get("output_tokens", 0)
                            if input_tokens:
                                turn_span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
                            if output_tokens:
                                turn_span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
                            cache_creation = message.usage.get("cache_creation_input_tokens", 0)
                            cache_read = message.usage.get("cache_read_input_tokens", 0)
                            if cache_creation:
                                turn_span.set_attribute("gen_ai.usage.cache_creation.input_tokens", cache_creation)
                            if cache_read:
                                turn_span.set_attribute("gen_ai.usage.cache_read.input_tokens", cache_read)
                        if hasattr(message, 'model') and message.model:
                            turn_span.set_attribute("gen_ai.response.model", message.model)

                        # Tool definitions from the proxy (available after first API request)
                        if self._compact_proxy and self._compact_proxy.last_tools:
                            turn_span.set_attribute(
                                "gen_ai.request.tools",
                                json.dumps(self._compact_proxy.last_tools),
                            )

                        # Close turn span BEFORE launching post-turn tasks.
                        # For wake-up turns, the span was __enter__'d in _run_session()'s
                        # context — if we create_task() while it's active, the new tasks
                        # inherit it as parent (context leakage). Closing first ensures
                        # suggest and archive run as root traces, not children.
                        turn_span.__exit__(None, None, None)
                        turn_span = None
                        self._turn_span = None
                        self._is_context_start = False

                        # Launch post-turn tasks (fire-and-forget, run as siblings)
                        if self._last_user_content and self._last_assistant_content:
                            async def _run_suggest():
                                try:
                                    memorables = await suggest(
                                        self._last_user_content,
                                        self._last_assistant_content,
                                        self._current_session_id or "unknown",
                                    )
                                    if memorables:
                                        self._pending_memorables.extend(memorables)
                                except Exception as e:
                                    logfire.error("Suggest failed: {error}", error=str(e))
                            self._suggest_task = asyncio.create_task(_run_suggest())

                        if self.archive and self._last_user_content:
                            asyncio.create_task(
                                archive_turn(
                                    user_content=self._last_user_content,
                                    assistant_content=self._last_assistant_content,
                                    session_id=self._current_session_id,
                                )
                            )

                    # If wake-up turn just completed, clear handoff state
                    if self._compact_mode == "waking_up":
                        self._compact_mode = None
                        self._needs_reorientation = False

                    # Session ID event
                    await self._push_event("session-id", {
                        "sessionId": message.session_id,
                    })

                    # Context event (token count)
                    if self._compact_proxy:
                        await self._push_event("context", {
                            "count": self._compact_proxy.token_count,
                            "window": self._compact_proxy.context_window,
                        })

                    # Turn-end event
                    await self._push_event("turn-end", {
                        "sessionId": message.session_id,
                        "cost": message.total_cost_usd,
                        "durationMs": message.duration_ms,
                    })

                    # Reset for next turn
                    assistant_text_parts = []

                    # Hand-off: send /compact through the generator
                    if self._pending_compact:
                        compact_instructions = self._pending_compact
                        self._pending_compact = None

                        # Open compact span (closed when wake-up is queued)
                        self._compact_span = logfire.span(
                            "compact", session_id=self._current_session_id,
                        )
                        self._compact_span.__enter__()
                        logfire.info(
                            "Compact: sending /compact ({chars} chars)",
                            chars=len(compact_instructions),
                        )

                        compact_cmd = f"/compact {compact_instructions}"
                        compact_message = {
                            "type": "user",
                            "message": {
                                "role": "user",
                                "content": [{"type": "text", "text": compact_cmd}],
                            },
                            "session_id": self._current_session_id or "new",
                        }

                        self._compact_mode = "compacting"
                        self._compact_summary_parts = []
                        self._compact_user_messages = []

                        # Tell the frontend we're compacting (keeps indicator alive)
                        await self._push_event("status", {"phase": "compacting"})

                        await self._queue.put(compact_message)

            # Wait for the query task to finish (generator exhausted or error)
            await query_task

        except Exception as e:
            await self._push_event("error", {"message": str(e)})
        finally:
            # Clean up compact span if still open (compaction interrupted)
            if self._compact_span is not None:
                self._compact_span.__exit__(None, None, None)
                self._compact_span = None
            # Clean up turn span if still open
            if turn_span is not None:
                turn_span.__exit__(None, None, None)
            # Clean up query task if still running
            if 'query_task' in locals() and not query_task.done():
                query_task.cancel()
            # Signal all subscribers to stop
            if self._broadcast:
                self._broadcast.close()

    async def _push_event(
        self,
        event_type: str | None,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Push an SSE event to all subscribers via broadcast.

        Args:
            event_type: Event type string, or None to close all subscribers
            data: Event data dict
        """
        if self._broadcast is None:
            return
        if event_type is None:
            self._broadcast.close()
            return
        self._broadcast.publish(event_type, data or {})

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def session_id(self) -> str | None:
        """Get the current session ID."""
        return self._current_session_id

    @property
    def connected(self) -> bool:
        """Check if the client is connected."""
        return self._sdk_client is not None

    @property
    def token_count(self) -> int:
        """Get the current token count (for context-o-meter)."""
        if self._compact_proxy:
            return self._compact_proxy.token_count
        return 0

    @property
    def context_window(self) -> int:
        """Get the context window size."""
        if self._compact_proxy:
            return self._compact_proxy.context_window
        return CompactProxy.DEFAULT_CONTEXT_WINDOW

    def _get_approach_light(self) -> str | None:
        """Check context usage and return approach light warning if threshold crossed."""
        if not self._compact_proxy:
            return None
        text, new_level = get_approach_light(
            self._compact_proxy.token_count,
            self._compact_proxy.context_window,
            self._approach_warned,
        )
        self._approach_warned = new_level
        return text

    @property
    def usage_7d(self) -> float | None:
        """Get the 7-day (weekly) usage as a float 0.0-1.0, or None if not yet known.

        Extracted from Anthropic response headers on every API call.
        Multiply by 100 for percentage.
        """
        if self._compact_proxy:
            return self._compact_proxy.usage_7d
        return None

    @property
    def usage_5h(self) -> float | None:
        """Get the 5-hour usage as a float 0.0-1.0, or None if not yet known."""
        if self._compact_proxy:
            return self._compact_proxy.usage_5h
        return None

    def set_token_count_callback(self, callback: "TokenCountCallback | None") -> None:
        """Set or replace the token count callback.

        Use this to swap in a per-turn callback that pushes events to
        the current SSE stream. The CompactProxy doesn't care what
        function it's holding—it just calls it when the count changes.
        """
        self._on_token_count = callback
        if self._compact_proxy:
            self._compact_proxy._on_token_count = callback

    def request_compact(self, instructions: str) -> None:
        """Flag a compact to fire after the current response stream finishes.

        Called by the hand-off MCP tool. The instructions get passed as
        the /compact argument and flow through to the summarizer.
        """
        self._pending_compact = instructions

    async def _store_memory_for_handoff(self, memory: str) -> dict:
        """Store a memory on behalf of the hand-off tool.

        The hand-off tool requires a last memory before lights-out.
        This wraps cortex.store() so the tool doesn't need to import it.
        """
        from .memories.cortex import store
        result = await store(memory)
        return result or {"id": "failed"}

    def clear_memorables(self) -> int:
        """Clear pending memorables and return how many were cleared.

        Used by Cortex store tool as feedback mechanism.
        """
        count = len(self._pending_memorables)
        self._pending_memorables = []
        return count

    # -------------------------------------------------------------------------
    # Image Processing
    # -------------------------------------------------------------------------

    async def _process_inline_images(self, content_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Process inline images in content blocks.

        For each image block:
        1. Resize to 768px long edge JPEG (safety valve + token efficiency)
        2. Save thumbnail to Alpha-Home/images/thumbnails/
        3. Replace original base64 with thumbnail base64
        4. Add a text block with the thumbnail path (sense memory nudge)
        5. Caption the image via Gemma 3 12B vision
        6. Search Cortex for related memories
        7. If matches found, inject text-only breadcrumbs (no image injection)

        This prevents Request Too Large errors from retina screenshots
        and makes every pasted image available for `cortex store --image`.
        """
        processed: list[dict[str, Any]] = []
        images_processed = 0

        for block in content_blocks:
            if block.get("type") == "image":
                source = block.get("source", {})
                if source.get("type") == "base64" and source.get("data"):
                    # Process the image
                    result = process_inline_image(
                        source["data"],
                        media_type=source.get("media_type", "image/png"),
                    )
                    if result:
                        new_base64, thumb_path = result
                        # Replace with resized image
                        processed.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",  # Always JPEG now
                                "data": new_base64,
                            },
                        })
                        # Sense memory nudge — remind Alpha to notice what she sees
                        processed.append({
                            "type": "text",
                            "text": f"📷 {thumb_path} — Remember this?",
                        })
                        images_processed += 1

                        # Image-triggered recall: caption → embed → search → breadcrumb
                        recall_text = await self._image_recall(new_base64)
                        if recall_text:
                            processed.append({
                                "type": "text",
                                "text": recall_text,
                            })

                        continue
                    # Fall through to original if processing fails

            processed.append(block)

        return processed

    async def _image_recall(self, base64_data: str) -> str | None:
        """Image-triggered memory recall.

        Pipeline: caption image → embed caption → search Cortex → format breadcrumbs.
        Returns formatted text block or None if no matches / pipeline fails.

        Graceful degradation: any failure returns None silently.
        The image still works normally — just the nudge, no recall.
        """
        try:
            # Caption the image via Gemma 3 12B vision
            caption = await caption_image(base64_data)
            if not caption:
                return None

            # Embed the caption text
            caption_embedding = await embed_query(caption)

            # Search Cortex with the caption embedding
            results = await search_memories(
                query_embedding=caption_embedding,
                query_text=caption,
                limit=3,
                min_score=0.5,
            )

            if not results:
                return None

            # Format as text-only breadcrumbs (no image injection)
            return format_image_recall(results)

        except EmbeddingError:
            return None
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    async def _build_orientation(self) -> list[dict[str, Any]]:
        """Build orientation blocks for session start."""
        return await build_orientation(self.client_name, self.hostname)

    async def _on_pre_compact(
        self,
        input: PreCompactHookInput,
        tool_use_id: str | None,
        context: HookContext,
    ) -> dict[str, Any]:
        """Hook called before compaction - flag that we need to re-orient on next turn."""
        self._needs_reorientation = True
        self._approach_warned = 0  # Reset approach lights after compaction
        return {"continue_": True}

    async def _create_sdk_client(self, session_id: str | None = None) -> None:
        """Create or recreate the SDK client."""
        # Disconnect existing client if any
        if self._sdk_client:
            await self._sdk_client.disconnect()

        # Build hooks config for PreCompact
        hooks = {
            "PreCompact": [
                HookMatcher(
                    matcher=None,  # Match all
                    hooks=[self._on_pre_compact],
                )
            ]
        }

        # Build MCP servers — internal + consumer-provided (consumer wins on conflict)
        internal_servers = {
            "cortex": create_cortex_server(
                get_session_id=lambda: self._current_session_id,
                clear_memorables=self.clear_memorables,
            ),
            "fetch": create_fetch_server(),
            "forge": create_forge_server(),
            "handoff": create_handoff_server(
                on_handoff=self.request_compact,
                store_memory=self._store_memory_for_handoff,
            ),
        }
        # Consumer-provided servers override internal ones with the same name
        merged_servers = {**internal_servers, **self.mcp_servers}

        # Build options with our system prompt
        options_kwargs = {
            "cwd": self.cwd,
            "system_prompt": self._system_prompt,  # Just the soul!
            "model": self.ALPHA_MODEL,  # Alpha IS this model
            "disallowed_tools": _SDK_DISALLOWED_TOOLS + (self.disallowed_tools or []),
            "mcp_servers": merged_servers,
            "include_partial_messages": self.include_partial_messages,
            "resume": session_id,
            "permission_mode": self.permission_mode,
            "hooks": hooks,
            # The Alpha Plugin — agents, skills, and MCP servers in one bundle
            "plugins": [{"type": "local", "path": _ALPHA_PLUGIN_DIR}],
            # Extended thinking — adaptive: think when it helps, don't when it doesn't
            "thinking": {"type": "adaptive"},
        }
        options = ClaudeAgentOptions(**options_kwargs)

        # Create and connect
        self._sdk_client = ClaudeSDKClient(options)
        await self._sdk_client.connect()

        self._current_session_id = session_id
