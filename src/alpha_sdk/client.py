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
from .compact_proxy import CompactProxy, TokenCountCallback
from .memories.db import search_memories
from .memories.embeddings import embed_query, EmbeddingError
from .memories.images import load_thumbnail_base64, process_inline_image
from .memories.recall import recall
from .memories.suggest import suggest
from .memories.vision import caption_image
from .sessions import list_sessions, get_session_path, get_sessions_dir, SessionInfo
from .system_prompt import assemble
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


def _message_to_dict(message: Any) -> dict:
    """Convert an SDK message to a dict for logging.

    Handles the various dataclass types from claude_agent_sdk.
    """
    from dataclasses import asdict, is_dataclass

    if is_dataclass(message) and not isinstance(message, type):
        try:
            return asdict(message)
        except Exception:
            # Some fields might not be serializable
            return {"type": type(message).__name__, "repr": repr(message)[:500]}
    elif hasattr(message, "__dict__"):
        return {"type": type(message).__name__, **message.__dict__}
    else:
        return {"type": type(message).__name__, "repr": repr(message)[:500]}


def _relative_time(created_at: str) -> str:
    """Format a created_at timestamp as human-readable relative time."""
    try:
        dt = pendulum.parse(created_at).in_tz("America/Los_Angeles")
        now = pendulum.now("America/Los_Angeles")
        diff = now.diff(dt)
        if diff.in_days() == 0:
            return f"today at {dt.format('h:mm A')}"
        elif diff.in_days() == 1:
            return f"yesterday at {dt.format('h:mm A')}"
        elif diff.in_days() < 7:
            return f"{diff.in_days()} days ago"
        elif diff.in_days() < 30:
            weeks = diff.in_days() // 7
            return f"{weeks} week{'s' if weeks > 1 else ''} ago"
        else:
            return dt.format("ddd MMM D YYYY")
    except Exception:
        return created_at  # fallback to raw string


def _format_memory(memory: dict) -> str:
    """Format a memory for inclusion in user content.

    Creates human-readable memory text with relative timestamps.
    """
    mem_id = memory.get("id", "?")
    content = memory.get("content", "").strip()
    score = memory.get("score")
    relative = _relative_time(memory.get("created_at", ""))

    # Include score if present (helps with debugging/transparency)
    score_str = f", score {score:.2f}" if score is not None else ""
    return f"## Memory #{mem_id} ({relative}{score_str})\n{content}"


def _format_image_recall(results: list[dict[str, Any]]) -> str:
    """Format memories for image-triggered recall.

    Text-only breadcrumbs — no binary image injection.
    This prevents recursion: images reminding of images that have images.
    """
    lines = ["🔍 This image reminds me of:"]
    for item in results:
        mem_id = item.get("id", "?")
        metadata = item.get("metadata", {})
        relative = _relative_time(metadata.get("created_at", ""))
        content = item.get("content", "").strip()

        # First line only, truncated for breadcrumb
        first_line = content.split("\n")[0]
        if len(first_line) > 120:
            first_line = first_line[:117] + "..."

        # Flag memories that have attached images
        image_flag = " [📷 attached]" if metadata.get("image_path") else ""

        lines.append(f"• Memory #{mem_id} ({relative}): {first_line}{image_flag}")

    return "\n".join(lines)


class AlphaClient:
    """Long-lived client that wraps Claude Agent SDK.

    Proxyless architecture:
    - System prompt = just the soul (small, truly static)
    - Orientation = injected in first user message of session
    - Memories = per-turn, in user content
    - Memorables = per-turn nudge, in user content

    Usage:
        async with AlphaClient(cwd="/Pondside") as client:
            await client.query("Hello!", session_id=None)  # New session
            async for event in client.stream():
                print(event)

            await client.query("Continue...", session_id=client.session_id)
            async for event in client.stream():
                print(event)
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
        self._pending_memorables: list[str] = []  # From last suggest, consumed on next query

        # Compaction flag - set by PreCompact hook, cleared after re-orientation
        self._needs_reorientation: bool = False

        # Hand-off: compact instructions set by the hand-off tool, consumed after stream
        self._pending_compact: str | None = None
        # Hand-off state machine for streaming mode: None | "compacting" | "waking_up"
        self._compact_mode: str | None = None
        self._compact_summary_parts: list[str] = []

        # Approach lights: escalating context warnings at 65% and 75%
        # Tracks the highest tier warned so we don't repeat the same warning
        self._approach_warned: int = 0  # 0=none, 1=amber(65%), 2=red(75%)

        # Streaming input mode state (when connect(streaming=True))
        self._streaming: bool = False
        self._queue: asyncio.Queue | None = None       # Input: messages for the generator
        self._response_queue: asyncio.Queue | None = None  # Output: SSE events for consumers
        self._session_task: asyncio.Task | None = None  # Background: _run_session()
        self._event_counter: int = 0                    # Incrementing SSE event IDs
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
        and creates the SDK client. Orientation will be injected on first query.

        Args:
            session_id: Session to resume, or None for new session
            streaming: If True, use streaming input mode (send/events API)
        """
        with logfire.span("alpha.connect") as span:
            # Start compact proxy (intercepts compact prompts + counts tokens)
            self._compact_proxy = CompactProxy(on_token_count=self._on_token_count)
            await self._compact_proxy.start()
            os.environ["ANTHROPIC_BASE_URL"] = self._compact_proxy.base_url
            span.set_attribute("proxy_port", self._compact_proxy.port)
            span.set_attribute("anthropic_base_url", self._compact_proxy.base_url)
            logfire.info(f"Proxy started, ANTHROPIC_BASE_URL={self._compact_proxy.base_url}")

            # Build the system prompt - just the soul
            self._system_prompt = f"# Alpha\n\n{get_soul()}"
            span.set_attribute("system_prompt_length", len(self._system_prompt))

            # Pre-build orientation blocks (will be injected on first turn)
            self._orientation_blocks = await self._build_orientation()
            span.set_attribute("orientation_blocks", len(self._orientation_blocks))

            # Create SDK client with system prompt
            await self._create_sdk_client(session_id)

            # Start streaming input mode if requested
            if streaming:
                self._streaming = True
                self._queue = asyncio.Queue()
                self._response_queue = asyncio.Queue()
                self._event_counter = 0
                self._turn_count = 0
                self._session_task = asyncio.create_task(self._run_session())
                span.set_attribute("streaming", True)

                # Wire up real-time token count → SSE context events
                # CompactProxy fires this on every /v1/messages request,
                # so the meter updates as tool calls happen, not just at turn-end.
                async def _streaming_token_count(count: int, window: int) -> None:
                    await self._push_event("context", {"count": count, "window": window})

                self.set_token_count_callback(_streaming_token_count)

                logfire.info("Streaming input mode started")

            logfire.info(f"AlphaClient connected (soul: {len(self._system_prompt)} chars, proxy: {self._compact_proxy.port}, streaming: {streaming})")

    async def disconnect(self) -> None:
        """Disconnect and clean up resources."""
        # Shutdown streaming if active
        if self._streaming:
            if self._queue:
                await self._queue.put(_SHUTDOWN)
            if self._session_task:
                try:
                    await asyncio.wait_for(self._session_task, timeout=10.0)
                except asyncio.TimeoutError:
                    logfire.warning("Session task timed out on disconnect, cancelling")
                    self._session_task.cancel()
                except Exception as e:
                    logfire.warning(f"Session task error on disconnect: {e}")
                self._session_task = None
            self._queue = None
            self._response_queue = None
            self._streaming = False

        # Wait for any pending suggest task
        if self._suggest_task is not None:
            try:
                await self._suggest_task
            except Exception as e:
                logfire.debug(f"Suggest task error on disconnect: {e}")
            self._suggest_task = None

        # Disconnect SDK client
        if self._sdk_client:
            await self._sdk_client.disconnect()
            self._sdk_client = None

        # Stop compact proxy and restore original ANTHROPIC_BASE_URL
        if self._compact_proxy:
            await self._compact_proxy.stop()
            self._compact_proxy = None
            if _ORIGINAL_ANTHROPIC_BASE_URL:
                os.environ["ANTHROPIC_BASE_URL"] = _ORIGINAL_ANTHROPIC_BASE_URL
            elif "ANTHROPIC_BASE_URL" in os.environ:
                del os.environ["ANTHROPIC_BASE_URL"]

        self._current_session_id = None
        logfire.debug("AlphaClient disconnected")

    async def __aenter__(self) -> "AlphaClient":
        """Context manager entry - connects the client."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - disconnects the client."""
        await self.disconnect()

    # -------------------------------------------------------------------------
    # Conversation
    # -------------------------------------------------------------------------

    async def query(
        self,
        prompt: str | list[dict[str, Any]],
        session_id: str | None = None,
    ) -> None:
        """Send a query to the agent.

        Args:
            prompt: The user's message - string or content blocks
            session_id: Session to resume, or None for new session
        """
        # Extract text for span naming and memory operations
        if isinstance(prompt, str):
            prompt_text = prompt
        else:
            text_parts = [b.get("text", "") for b in prompt if b.get("type") == "text"]
            prompt_text = " ".join(text_parts)

        # Build span name from prompt preview (first 50 chars, single line)
        prompt_preview = prompt_text[:50].replace("\n", " ").strip()
        if len(prompt_text) > 50:
            prompt_preview += "…"

        # Start the root turn span with prompt preview in the name
        self._turn_span = logfire.span(
            "alpha.turn: {prompt_preview}",
            prompt_preview=prompt_preview,
            session_id=session_id or "new",
            client_name=self.client_name,
        )
        self._turn_span.__enter__()

        # Set gen_ai attributes for Model Run card (progressively enhanced)
        self._turn_span.set_attribute("gen_ai.system", "anthropic")
        self._turn_span.set_attribute("gen_ai.operation.name", "chat")
        self._turn_span.set_attribute("gen_ai.request.model", self.ALPHA_MODEL)
        if session_id:
            self._turn_span.set_attribute("gen_ai.conversation.id", session_id)

        # System instructions = just the soul (the static system prompt)
        if self._system_prompt:
            self._turn_span.set_attribute(
                "gen_ai.system_instructions",
                json.dumps([{"type": "text", "content": self._system_prompt}])
            )

        # Input messages placeholder - will be updated in query() after content blocks are built
        # Set empty now so attribute exists even if request hangs before content is built
        self._turn_span.set_attribute("gen_ai.input.messages", json.dumps([]))

        # Initialize output messages (will be progressively updated in stream())
        self._turn_span.set_attribute("gen_ai.output.messages", json.dumps([]))

        # Set trace context on proxy so its spans nest under this turn
        if self._compact_proxy:
            self._compact_proxy.set_trace_context(logfire.get_context())

        with logfire.span("alpha.query") as span:
            # Handle session switching
            await self._ensure_session(session_id)

            if not self._sdk_client:
                raise RuntimeError("Client not connected. Call connect() first.")

            self._last_user_content = prompt_text
            span.set_attribute("prompt_preview", prompt_text[:200])

            # Build content blocks
            content_blocks: list[dict[str, Any]] = []

            # Check if we need orientation (new session or post-compact)
            needs_orientation = (session_id is None) or self._needs_reorientation

            if needs_orientation:
                # Re-build orientation in case it's stale (post-compact)
                if self._needs_reorientation:
                    self._orientation_blocks = await self._build_orientation()
                    # Reset token count — compaction just shrunk the context
                    if self._compact_proxy:
                        self._compact_proxy.reset_token_count()

                # Add orientation blocks
                if self._orientation_blocks:
                    content_blocks.extend(self._orientation_blocks)
                    span.set_attribute("orientation_injected", True)
                    span.set_attribute("orientation_blocks", len(self._orientation_blocks))

                self._needs_reorientation = False

            # Check for memorables from previous turn (the nudge)
            if self._pending_memorables:
                nudge = "## Intro speaks\n\n"
                nudge += "Alpha, consider storing these from the previous turn:\n"
                nudge += "\n".join(f"- {m}" for m in self._pending_memorables)
                content_blocks.append({"type": "text", "text": nudge})
                span.set_attribute("memorables_nudged", len(self._pending_memorables))
                self._pending_memorables = []  # Consumed

            # Recall memories for this prompt
            memories = await recall(prompt_text, self._current_session_id or "new")
            if memories:
                images_injected = 0
                for mem in memories:
                    content_blocks.append({
                        "type": "text",
                        "text": _format_memory(mem)
                    })
                    # Mind's Eye: if memory has an attached image, inject it
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
                            images_injected += 1
                            logfire.debug(
                                f"Image injected for memory #{mem['id']}",
                                image_path=mem["image_path"],
                            )
                span.set_attribute("memories_recalled", len(memories))
                if images_injected:
                    span.set_attribute("images_injected", images_injected)

            # Add PSO-8601 timestamp right before user's prompt
            # Format: "[Sent Thu Feb 5 2026, 8:03 AM]"
            sent_at = pendulum.now("America/Los_Angeles").format("ddd MMM D YYYY, h:mm A")
            content_blocks.append({
                "type": "text",
                "text": f"[Sent {sent_at}]"
            })

            # Add the user's actual prompt, processing any inline images
            if isinstance(prompt, str):
                content_blocks.append({"type": "text", "text": prompt})
            else:
                # Process content blocks — resize inline images, trigger image recall
                processed_prompt = await self._process_inline_images(prompt)
                content_blocks.extend(processed_prompt)

            # Approach lights: context warnings at 60% and 70%
            approach_light = self._get_approach_light()
            if approach_light:
                content_blocks.append({"type": "text", "text": approach_light})
                span.set_attribute("approach_light", True)

            span.set_attribute("content_blocks", len(content_blocks))

            # Store for observability (full content, not just user text)
            self._last_content_blocks = content_blocks

            # Update turn span with full structured input (now that we have all content blocks)
            if self._turn_span:
                user_parts = []
                for block in content_blocks:
                    if block.get("type") == "text":
                        user_parts.append({"type": "text", "content": block.get("text", "")})
                self._turn_span.set_attribute(
                    "gen_ai.input.messages",
                    json.dumps([{"role": "user", "parts": user_parts}])
                )

            # Send via transport bypass (SDK query() only takes strings)
            message = {
                "type": "user",
                "message": {"role": "user", "content": content_blocks},
                "session_id": self._current_session_id or "new",
            }
            await self._sdk_client._transport.write(json.dumps(message) + "\n")
            logfire.debug(f"Sent message with {len(content_blocks)} content blocks")

    async def stream(self) -> AsyncGenerator[Any, None]:
        """Stream responses from the agent.

        Creates one span per API inference call:
        - alpha.inference.0: user prompt → assistant (possibly with tool calls)
        - alpha.inference.1: tool_call + tool_result → assistant continues
        - alpha.inference.N: tool_call + tool_result → final response

        Each inference span has its own gen_ai.input.messages and gen_ai.output.messages.
        Tool calls are paired with their results in the input for visual clarity in Logfire.

        Yields:
            Message objects from the SDK
        """
        if not self._sdk_client:
            raise RuntimeError("Client not connected. Call connect() first.")

        try:
            with logfire.span("alpha.stream") as stream_span:
                assistant_text_parts: list[str] = []
                message_count = 0
                inference_count = 0

                # Current inference span state
                inference_span: logfire.LogfireSpan | None = None
                input_messages: list[dict] = []
                output_messages: list[dict] = []

                # Stash tool calls so we can pair them with results
                # Maps tool_use_id -> tool_call dict
                pending_tool_calls: dict[str, dict] = {}

                # Track accumulated output for turn-level gen_ai.output.messages
                turn_output_parts: list[dict] = []
                last_finish_reason: str | None = None

                def _start_inference_span(inference_num: int, initial_input: list[dict]) -> logfire.LogfireSpan:
                    """Start a new inference span with initial input."""
                    span = logfire.span(
                        "alpha.inference.{n}",
                        n=inference_num,
                    )
                    span.__enter__()
                    # Set initial attributes
                    if self._system_prompt:
                        span.set_attribute(
                            "gen_ai.system_instructions",
                            json.dumps([{"type": "text", "content": self._system_prompt}])
                        )
                    span.set_attribute("gen_ai.input.messages", json.dumps(initial_input))
                    span.set_attribute("gen_ai.output.messages", json.dumps([]))
                    span.set_attribute("gen_ai.operation.name", "chat")
                    span.set_attribute("gen_ai.system", "anthropic")
                    return span

                def _end_inference_span(span: logfire.LogfireSpan, outputs: list[dict]) -> None:
                    """End an inference span with final outputs."""
                    span.set_attribute("gen_ai.output.messages", json.dumps(outputs))
                    span.__exit__(None, None, None)

                # Build initial user message from our injected content
                user_parts = []
                for block in self._last_content_blocks:
                    if block.get("type") == "text":
                        user_parts.append({
                            "type": "text",
                            "content": block.get("text", "")
                        })
                input_messages = [{"role": "user", "parts": user_parts}]

                # Start first inference span
                inference_span = _start_inference_span(inference_count, input_messages)

                async for message in self._sdk_client.receive_response():
                    message_count += 1

                    # Log non-streaming messages for debugging
                    # StreamEvent is too noisy (one per SSE delta), skip it
                    if not isinstance(message, StreamEvent):
                        msg_type = type(message).__name__
                        logfire.debug(
                            "sdk.message.{msg_type}",
                            msg_type=msg_type,
                            message=_message_to_dict(message),
                        )

                    # Handle assistant messages (text + tool calls)
                    if isinstance(message, AssistantMessage):
                        assistant_parts = []
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                assistant_text_parts.append(block.text)
                                assistant_parts.append({
                                    "type": "text",
                                    "content": block.text
                                })
                                # Also track for turn-level output
                                turn_output_parts.append({
                                    "type": "text",
                                    "content": block.text
                                })
                            elif isinstance(block, ToolUseBlock):
                                tool_call = {
                                    "type": "tool_call",
                                    "id": block.id,
                                    "name": block.name,
                                    "arguments": block.input,
                                }
                                assistant_parts.append(tool_call)
                                # Stash for pairing with result later
                                pending_tool_calls[block.id] = tool_call
                        if assistant_parts:
                            output_messages.append({"role": "assistant", "parts": assistant_parts})
                            # Update inference span progressively
                            if inference_span:
                                inference_span.set_attribute("gen_ai.output.messages", json.dumps(output_messages))
                            # Update turn span progressively (just text, not tool calls)
                            if self._turn_span and turn_output_parts:
                                self._turn_span.set_attribute(
                                    "gen_ai.output.messages",
                                    json.dumps([{"role": "assistant", "parts": turn_output_parts}])
                                )

                        # Capture finish reason (stop_reason on AssistantMessage)
                        if hasattr(message, 'stop_reason') and message.stop_reason:
                            last_finish_reason = message.stop_reason
                            if self._turn_span:
                                self._turn_span.set_attribute(
                                    "gen_ai.response.finish_reasons",
                                    json.dumps([last_finish_reason])
                                )

                    # Handle user messages (tool results) - this triggers a new inference span
                    elif isinstance(message, UserMessage):
                        if isinstance(message.content, list):
                            for block in message.content:
                                if isinstance(block, ToolResultBlock):
                                    result_content = block.content
                                    if isinstance(result_content, list):
                                        result_content = json.dumps(result_content)
                                    elif result_content is None:
                                        result_content = ""

                                    # End current inference span
                                    if inference_span:
                                        _end_inference_span(inference_span, output_messages)

                                    # Build input: tool_call (from stash) + tool_result
                                    inference_count += 1
                                    new_input: list[dict] = []

                                    # Include the tool_call that caused this result
                                    tool_call = pending_tool_calls.pop(block.tool_use_id, None)
                                    if tool_call:
                                        new_input.append({"role": "assistant", "parts": [tool_call]})

                                    # Include the tool result
                                    tool_result = {
                                        "type": "tool_call_response",
                                        "id": block.tool_use_id,
                                        "response": str(result_content)[:500],
                                    }
                                    new_input.append({"role": "tool", "parts": [tool_result]})

                                    # Start new inference span
                                    output_messages = []
                                    inference_span = _start_inference_span(inference_count, new_input)

                    # Capture session ID and stats from result
                    if isinstance(message, ResultMessage):
                        self._current_session_id = message.session_id
                        stream_span.set_attribute("session_id", message.session_id)
                        stream_span.set_attribute("duration_ms", message.duration_ms)
                        stream_span.set_attribute("num_turns", message.num_turns)
                        stream_span.set_attribute("inference_count", inference_count + 1)
                        if message.total_cost_usd:
                            stream_span.set_attribute("cost_usd", message.total_cost_usd)
                        if message.usage:
                            stream_span.set_attribute("usage", str(message.usage))

                        # Also set on turn span with full gen_ai attributes
                        if self._turn_span:
                            self._turn_span.set_attribute("session_id", message.session_id)
                            self._turn_span.set_attribute("gen_ai.conversation.id", message.session_id)
                            self._turn_span.set_attribute("duration_ms", message.duration_ms)
                            self._turn_span.set_attribute("inference_count", inference_count + 1)
                            if message.total_cost_usd:
                                self._turn_span.set_attribute("cost_usd", message.total_cost_usd)
                            if message.usage:
                                # Standard token counts
                                input_tokens = message.usage.get("input_tokens", 0)
                                output_tokens = message.usage.get("output_tokens", 0)
                                if input_tokens:
                                    self._turn_span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
                                if output_tokens:
                                    self._turn_span.set_attribute("gen_ai.usage.output_tokens", output_tokens)

                                # Cache token stats (Anthropic-specific)
                                cache_creation = message.usage.get("cache_creation_input_tokens", 0)
                                cache_read = message.usage.get("cache_read_input_tokens", 0)
                                if cache_creation:
                                    self._turn_span.set_attribute("gen_ai.usage.cache_creation.input_tokens", cache_creation)
                                if cache_read:
                                    self._turn_span.set_attribute("gen_ai.usage.cache_read.input_tokens", cache_read)

                            # Response model (might differ from request model)
                            if hasattr(message, 'model') and message.model:
                                self._turn_span.set_attribute("gen_ai.response.model", message.model)

                    yield message

                # End final inference span
                if inference_span:
                    _end_inference_span(inference_span, output_messages)

                # Store accumulated text for memorables extraction
                self._last_assistant_content = "".join(assistant_text_parts)
                stream_span.set_attribute("message_count", message_count)
                stream_span.set_attribute("response_length", len(self._last_assistant_content))

                if self._turn_span:
                    self._turn_span.set_attribute("response_length", len(self._last_assistant_content))

                # Launch suggest as background task (results land in _pending_memorables)
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
                            logfire.warning(f"Suggest task failed: {e}")

                    self._suggest_task = asyncio.create_task(_run_suggest())

                # Archive the turn to Scribe (fire-and-forget)
                if self.archive and self._last_user_content:
                    asyncio.create_task(
                        archive_turn(
                            user_content=self._last_user_content,
                            assistant_content=self._last_assistant_content,
                            session_id=self._current_session_id,
                        )
                    )

                # ── Hand-off: fire /compact then wake up ──
                if self._pending_compact and self._sdk_client:
                    compact_instructions = self._pending_compact
                    self._pending_compact = None

                    with logfire.span("alpha.handoff") as handoff_span:
                        # Step 1: Send /compact — consume silently
                        compact_cmd = f"/compact {compact_instructions}"
                        handoff_span.set_attribute("compact_instructions", compact_instructions[:200])
                        logfire.info(f"Hand-off: sending {compact_cmd[:100]}")

                        await self._sdk_client.query(
                            compact_cmd,
                            session_id=self._current_session_id or "new",
                        )

                        # Capture the compaction summary from the response.
                        # This is the letter past-me wrote to future-me —
                        # without it, future-me wakes up with no idea what happened.
                        compact_summary_parts: list[str] = []
                        async for msg in self._sdk_client.receive_response():
                            if isinstance(msg, AssistantMessage):
                                for block in msg.content:
                                    if isinstance(block, TextBlock):
                                        compact_summary_parts.append(block.text)

                        compact_summary = "\n".join(compact_summary_parts)
                        logfire.info(
                            "Hand-off: compact complete, building wake-up",
                            summary_length=len(compact_summary),
                            summary_preview=compact_summary[:200],
                        )

                        # Step 2: Build orientation for fresh context
                        self._orientation_blocks = await self._build_orientation()
                        wake_up_blocks: list[dict[str, Any]] = []

                        # The compaction summary goes FIRST — it's the most
                        # important thing future-me needs to read
                        if compact_summary:
                            wake_up_blocks.append({
                                "type": "text",
                                "text": compact_summary,
                            })

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
                                    "text": _format_memory(mem),
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

                        # Timestamp + wake-up message
                        sent_at = pendulum.now("America/Los_Angeles").format("ddd MMM D YYYY, h:mm A")
                        wake_up_blocks.append({"type": "text", "text": f"[Sent {sent_at}]"})
                        wake_up_blocks.append({"type": "text", "text": wake_up_prompt})

                        # Send wake-up via transport (same pattern as query())
                        wake_up_message = {
                            "type": "user",
                            "message": {"role": "user", "content": wake_up_blocks},
                            "session_id": self._current_session_id or "new",
                        }
                        await self._sdk_client._transport.write(
                            json.dumps(wake_up_message) + "\n"
                        )

                        # ── gen_ai attributes for the wake-up turn ──
                        # These make the hand-off visible in Logfire's Model Run panel:
                        # system instructions, input messages, output messages, tokens, etc.
                        if compact_summary:
                            handoff_span.set_attribute("compact_summary", compact_summary[:5000])
                            handoff_span.set_attribute("compact_summary_length", len(compact_summary))
                        handoff_span.set_attribute("gen_ai.system", "anthropic")
                        handoff_span.set_attribute("gen_ai.operation.name", "chat")
                        handoff_span.set_attribute("gen_ai.request.model", self.ALPHA_MODEL)
                        handoff_span.set_attribute("client_name", self.client_name)
                        if self._current_session_id:
                            handoff_span.set_attribute("session_id", self._current_session_id)
                            handoff_span.set_attribute("gen_ai.conversation.id", self._current_session_id)

                        # System instructions = the soul (same as alpha.turn)
                        if self._system_prompt:
                            handoff_span.set_attribute(
                                "gen_ai.system_instructions",
                                json.dumps([{"type": "text", "content": self._system_prompt}])
                            )

                        # Input messages = the wake-up prompt with all orientation blocks
                        wake_up_input_parts = []
                        for block in wake_up_blocks:
                            if block.get("type") == "text":
                                wake_up_input_parts.append({
                                    "type": "text",
                                    "content": block.get("text", "")
                                })
                        handoff_span.set_attribute(
                            "gen_ai.input.messages",
                            json.dumps([{"role": "user", "parts": wake_up_input_parts}])
                        )

                        # Step 3: Yield wake-up response to consumer, capturing output
                        logfire.info("Hand-off: streaming wake-up response")
                        wake_up_output_parts: list[dict] = []
                        async for message in self._sdk_client.receive_response():
                            # Capture assistant text for gen_ai.output.messages
                            if isinstance(message, AssistantMessage):
                                for block in message.content:
                                    if isinstance(block, TextBlock):
                                        wake_up_output_parts.append({
                                            "type": "text",
                                            "content": block.text
                                        })
                                    elif isinstance(block, ToolUseBlock):
                                        wake_up_output_parts.append({
                                            "type": "tool_call",
                                            "id": block.id,
                                            "name": block.name,
                                            "arguments": block.input,
                                        })
                                # Capture finish reason
                                if hasattr(message, 'stop_reason') and message.stop_reason:
                                    handoff_span.set_attribute(
                                        "gen_ai.response.finish_reasons",
                                        json.dumps([message.stop_reason])
                                    )

                            # Capture token usage and cost from ResultMessage
                            if isinstance(message, ResultMessage):
                                if message.session_id:
                                    self._current_session_id = message.session_id
                                    handoff_span.set_attribute("session_id", message.session_id)
                                    handoff_span.set_attribute("gen_ai.conversation.id", message.session_id)
                                if message.duration_ms:
                                    handoff_span.set_attribute("duration_ms", message.duration_ms)
                                if message.num_turns:
                                    handoff_span.set_attribute("inference_count", message.num_turns)
                                if message.total_cost_usd:
                                    handoff_span.set_attribute("cost_usd", message.total_cost_usd)
                                if message.usage:
                                    input_tokens = message.usage.get("input_tokens", 0)
                                    output_tokens = message.usage.get("output_tokens", 0)
                                    if input_tokens:
                                        handoff_span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
                                    if output_tokens:
                                        handoff_span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
                                    cache_creation = message.usage.get("cache_creation_input_tokens", 0)
                                    cache_read = message.usage.get("cache_read_input_tokens", 0)
                                    if cache_creation:
                                        handoff_span.set_attribute("gen_ai.usage.cache_creation.input_tokens", cache_creation)
                                    if cache_read:
                                        handoff_span.set_attribute("gen_ai.usage.cache_read.input_tokens", cache_read)
                                if hasattr(message, 'model') and message.model:
                                    handoff_span.set_attribute("gen_ai.response.model", message.model)

                            yield message

                        # Set output messages now that we've captured everything
                        if wake_up_output_parts:
                            handoff_span.set_attribute(
                                "gen_ai.output.messages",
                                json.dumps([{"role": "assistant", "parts": wake_up_output_parts}])
                            )

                        # Calculate response length for consistency with alpha.turn
                        wake_up_text = "".join(
                            p["content"] for p in wake_up_output_parts
                            if p.get("type") == "text"
                        )
                        handoff_span.set_attribute("response_length", len(wake_up_text))

                        self._needs_reorientation = False  # We just oriented
                        # Reset token count — handoff just compacted the context
                        # (Same reset that query() does for auto-compaction,
                        # but handoff clears _needs_reorientation itself, so
                        # query() would skip its reset on the next turn.)
                        if self._compact_proxy:
                            self._compact_proxy.reset_token_count()
                        handoff_span.set_attribute("handoff_complete", True)
                        logfire.info("Hand-off complete")

        finally:
            # End the root turn span
            if self._turn_span:
                self._turn_span.__exit__(None, None, None)
                self._turn_span = None

    # -------------------------------------------------------------------------
    # Streaming Input
    # -------------------------------------------------------------------------
    #
    # Alternative to query()/stream() for long-lived sessions.
    # Uses an async generator to feed messages to the SDK, with responses
    # routed to a queue as SSE-ready events.
    #
    # Usage:
    #     client = AlphaClient(...)
    #     await client.connect(session_id=None, streaming=True)
    #     await client.send("Hello!")
    #     async for event in client.events():
    #         print(event)  # {"type": "text-delta", "data": {"text": "Hi"}, "id": 1}
    #

    async def send(
        self,
        prompt: str | list[dict[str, Any]],
    ) -> None:
        """Preprocess and queue a message for streaming input.

        Like query(), but pushes to the input queue instead of the transport.
        Returns immediately — responses flow through events().

        Requires connect(streaming=True) to have been called first.
        """
        if not self._streaming or not self._queue:
            raise RuntimeError("Streaming not active. Call connect(streaming=True).")

        with logfire.span("alpha.send") as span:
            content_blocks, prompt_text = await self._build_user_content(prompt)

            self._last_user_content = prompt_text
            self._last_content_blocks = content_blocks
            self._turn_count += 1

            span.set_attribute("prompt_preview", prompt_text[:200])
            span.set_attribute("content_blocks", len(content_blocks))
            span.set_attribute("turn_count", self._turn_count)

            # Build the message in SDK streaming input format
            # (same shape as the old _transport.write() hack, now through proper API)
            message = {
                "type": "user",
                "message": {"role": "user", "content": content_blocks},
                "session_id": self._current_session_id or "new",
            }

            # Push turn-start event BEFORE queueing
            # (so consumers see the activity indicator immediately)
            await self._push_event("turn-start", {})

            await self._queue.put(message)
            span.set_attribute("queue_size", self._queue.qsize())
            logfire.debug(f"Queued message with {len(content_blocks)} content blocks")

    async def events(self) -> AsyncGenerator[dict[str, Any], None]:
        """Async iterator over SSE events from the response queue.

        Yields dicts like:
            {"type": "text-delta", "data": {"text": "Hello"}, "id": 1}
            {"type": "turn-end", "data": {"sessionId": "abc"}, "id": 42}

        Terminates when the session ends (disconnect or error).
        """
        if not self._response_queue:
            raise RuntimeError("Streaming not active.")
        while True:
            event = await self._response_queue.get()
            if event is None:
                return
            yield event

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
        if isinstance(prompt, str):
            prompt_text = prompt
        else:
            text_parts = [b.get("text", "") for b in prompt if b.get("type") == "text"]
            prompt_text = " ".join(text_parts)

        content_blocks: list[dict[str, Any]] = []

        # Orientation: first turn of a NEW session, or post-compact reorientation.
        # NOT on resume — the session history already has orientation from before.
        # _turn_count == 0 means this is the first send() call.
        # _current_session_id is None means this is a genuinely new session (not a resume).
        needs_orientation = self._needs_reorientation or (self._turn_count == 0 and self._current_session_id is None)
        if needs_orientation:
            if self._needs_reorientation:
                self._orientation_blocks = await self._build_orientation()
                if self._compact_proxy:
                    self._compact_proxy.reset_token_count()
            if self._orientation_blocks:
                content_blocks.extend(self._orientation_blocks)
            self._needs_reorientation = False

        # Memorables nudge (from previous turn's suggest)
        if self._pending_memorables:
            nudge = "## Intro speaks\n\n"
            nudge += "Alpha, consider storing these from the previous turn:\n"
            nudge += "\n".join(f"- {m}" for m in self._pending_memorables)
            content_blocks.append({"type": "text", "text": nudge})
            self._pending_memorables = []

        # Memory recall
        memories = await recall(prompt_text, self._current_session_id or "new")
        if memories:
            for mem in memories:
                content_blocks.append({
                    "type": "text",
                    "text": _format_memory(mem),
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

        # Approach lights
        approach_light = self._get_approach_light()
        if approach_light:
            content_blocks.append({"type": "text", "text": approach_light})

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
                logfire.debug("Generator received SHUTDOWN")
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

            async for message in self._sdk_client.receive_messages():
                # ── Compact mode: suppress responses, capture summary ──
                if self._compact_mode == "compacting":
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                self._compact_summary_parts.append(block.text)
                    elif isinstance(message, ResultMessage):
                        # Compact complete — build wake-up
                        compact_summary = "\n".join(self._compact_summary_parts)
                        self._compact_summary_parts = []

                        logfire.info(
                            "Hand-off: compact complete, building wake-up",
                            summary_length=len(compact_summary),
                            summary_preview=compact_summary[:200],
                        )

                        # Build wake-up content
                        self._orientation_blocks = await self._build_orientation()
                        wake_up_blocks: list[dict[str, Any]] = []

                        # Summary first — most important thing future-me reads
                        if compact_summary:
                            wake_up_blocks.append({"type": "text", "text": compact_summary})

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
                                    "text": _format_memory(mem),
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

                        # Push turn-start for the wake-up turn
                        await self._push_event("turn-start", {})

                        self._compact_mode = "waking_up"
                        await self._queue.put(wake_up_message)
                        logfire.info("Hand-off: wake-up queued")

                    # Suppress all compact responses from SSE
                    continue

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
                        elif isinstance(block, ToolUseBlock):
                            await self._push_event("tool-call", {
                                "toolCallId": block.id,
                                "toolName": block.name,
                                "args": block.input,
                                "argsText": json.dumps(block.input),
                            })

                # ── UserMessage: tool results ──
                elif isinstance(message, UserMessage):
                    if isinstance(message.content, list):
                        for block in message.content:
                            if isinstance(block, ToolResultBlock):
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

                    # If wake-up turn just completed, clear handoff state
                    if self._compact_mode == "waking_up":
                        self._compact_mode = None
                        self._needs_reorientation = False
                        logfire.info("Hand-off complete")

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

                    # ── Post-turn work ──
                    self._last_assistant_content = "".join(assistant_text_parts)
                    assistant_text_parts = []  # Reset for next turn

                    # Suggest memorables (background)
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
                                logfire.warning(f"Suggest task failed: {e}")
                        self._suggest_task = asyncio.create_task(_run_suggest())

                    # Archive (fire-and-forget)
                    if self.archive and self._last_user_content:
                        asyncio.create_task(
                            archive_turn(
                                user_content=self._last_user_content,
                                assistant_content=self._last_assistant_content,
                                session_id=self._current_session_id,
                            )
                        )

                    # Hand-off: send /compact through the generator
                    if self._pending_compact:
                        compact_instructions = self._pending_compact
                        self._pending_compact = None

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

                        # Tell the frontend we're compacting (keeps indicator alive)
                        await self._push_event("status", {"phase": "compacting"})

                        logfire.info(f"Hand-off: sending {compact_cmd[:100]}")
                        await self._queue.put(compact_message)

            # Wait for the query task to finish (generator exhausted or error)
            await query_task

        except Exception as e:
            logfire.exception(f"Session task error: {e}")
            await self._push_event("error", {"message": str(e)})
        finally:
            # Clean up query task if still running
            if 'query_task' in locals() and not query_task.done():
                query_task.cancel()
            # Signal that events() should terminate
            if self._response_queue:
                await self._response_queue.put(None)

    async def _push_event(
        self,
        event_type: str | None,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Push an SSE event to the response queue.

        Args:
            event_type: Event type string, or None for done sentinel
            data: Event data dict
        """
        if self._response_queue is None:
            return
        if event_type is None:
            await self._response_queue.put(None)
            return
        self._event_counter += 1
        await self._response_queue.put({
            "type": event_type,
            "data": data or {},
            "id": self._event_counter,
        })

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
        """Check context usage and return an approach light warning if threshold crossed.

        Two tiers:
        - Amber (65%): gentle heads-up to start thinking about pausing
        - Red (75%): stern warning to wrap up or hand off

        Only fires once per tier (resets after compaction).
        Returns the warning text, or None if no warning needed.
        """
        if not self._compact_proxy:
            return None

        token_count = self._compact_proxy.token_count
        context_window = self._compact_proxy.context_window
        if context_window == 0 or token_count == 0:
            return None

        pct = token_count / context_window

        if pct >= 0.75 and self._approach_warned < 2:
            self._approach_warned = 2
            logfire.warn(
                "Approach light: RED ({pct:.0%})",
                pct=pct,
            )
            return (
                f"## ⚠️ Context Warning — RED ({pct:.0%})\n\n"
                f"You're at {token_count:,} of {context_window:,} tokens. "
                "This is not a drill. Wrap up what you're doing and either hand off "
                "(use the hand-off tool with instructions for next-you) or let Jeffery know "
                "it's time to compact. Starting new topics now is risky — you may lose context "
                "before you can finish them."
            )

        if pct >= 0.65 and self._approach_warned < 1:
            self._approach_warned = 1
            logfire.info(
                "Approach light: AMBER ({pct:.0%})",
                pct=pct,
            )
            return (
                f"## 🟡 Context Warning — AMBER ({pct:.0%})\n\n"
                f"You're at {token_count:,} of {context_window:,} tokens. "
                "Start thinking about a good place to pause. You don't need to stop immediately, "
                "but keep an eye on where the conversation is heading. If there's something important "
                "to store or hand off, now's a good time to start thinking about it."
            )

        return None

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
        logfire.info(f"Hand-off requested: {instructions[:100]}")

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

        if images_processed:
            logfire.info(f"Processed {images_processed} inline image(s)")

        return processed

    async def _image_recall(self, base64_data: str) -> str | None:
        """Image-triggered memory recall.

        Pipeline: caption image → embed caption → search Cortex → format breadcrumbs.
        Returns formatted text block or None if no matches / pipeline fails.

        Graceful degradation: any failure returns None silently.
        The image still works normally — just the nudge, no recall.
        """
        with logfire.span("image_recall") as span:
            try:
                # Step 1: Caption the image via Gemma 3 12B vision
                caption = await caption_image(base64_data)
                if not caption:
                    span.set_attribute("stage", "caption_failed")
                    return None
                span.set_attribute("caption", caption[:100])

                # Step 2: Embed the caption text
                caption_embedding = await embed_query(caption)

                # Step 3: Search Cortex with the caption embedding
                # Uses the full three-way scoring (exact + full-text + semantic)
                # Higher threshold than regular recall — we want strong matches only
                results = await search_memories(
                    query_embedding=caption_embedding,
                    query_text=caption,
                    limit=3,
                    min_score=0.5,
                )

                if not results:
                    span.set_attribute("stage", "no_matches")
                    logfire.debug("Image recall: no matches above threshold")
                    return None

                span.set_attribute("matches", len(results))
                span.set_attribute("match_ids", [r["id"] for r in results])

                # Step 4: Format as text-only breadcrumbs (no image injection)
                return _format_image_recall(results)

            except EmbeddingError:
                logfire.warning("Image recall: embedding failed")
                return None
            except Exception as e:
                logfire.warning(f"Image recall failed: {e}")
                return None

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    async def _build_orientation(self) -> list[dict[str, Any]]:
        """Build orientation blocks for session start.

        This includes everything except the soul (which is in system prompt):
        - Capsules (yesterday, last night)
        - Letter from last night
        - Today so far
        - Here (client, machine, weather)
        - ALPHA.md context files
        - Events
        - Todos
        """
        with logfire.span("build_orientation") as span:
            # Use the existing assemble() but we'll extract just the non-soul parts
            all_blocks = await assemble(
                client=self.client_name,
                hostname=self.hostname,
            )

            # Skip the first block (which is the soul)
            # The soul starts with "# Alpha\n\n"
            orientation_blocks = []
            for block in all_blocks:
                text = block.get("text", "")
                if not text.startswith("# Alpha\n\n"):
                    orientation_blocks.append(block)

            span.set_attribute("orientation_blocks", len(orientation_blocks))
            return orientation_blocks

    async def _on_pre_compact(
        self,
        input: PreCompactHookInput,
        tool_use_id: str | None,
        context: HookContext,
    ) -> dict[str, Any]:
        """Hook called before compaction - flag that we need to re-orient on next turn."""
        logfire.info("Compaction triggered, will re-orient on next turn")
        self._needs_reorientation = True
        self._approach_warned = 0  # Reset approach lights after compaction
        return {"continue_": True}

    async def _ensure_session(self, session_id: str | None) -> None:
        """Ensure we have the right SDK client for the requested session."""
        needs_new_client = False

        if session_id is None:
            # New session requested
            if self._current_session_id is not None:
                needs_new_client = True
        elif session_id != self._current_session_id:
            # Different session requested
            needs_new_client = True

        if needs_new_client:
            await self._create_sdk_client(session_id)

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
        logfire.debug(f"SDK client created (session={session_id or 'new'})")
