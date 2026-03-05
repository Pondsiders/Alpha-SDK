"""test_claude.py — Tests for the Claude class.

Tests the wire protocol contract: event parsing, message formatting,
and state machine guards. No subprocess, no HTTP — pure logic.

What's NOT here (and why):
- Config storage tests ("does constructor store model?") — trust Python.
- Proxy property defaults — tested implicitly by integration tests.
- Dataclass field existence — trust the dataclass decorator.
"""

import pytest

from alpha_sdk.claude import (
    Claude,
    ClaudeState,
    Event,
    InitEvent,
    AssistantEvent,
    ResultEvent,
    StreamEvent,
    SystemEvent,
    ErrorEvent,
    _ControlRequestEvent,
)


# -- Event Parsing: Wire Protocol Contract -----------------------------------
#
# Claude._parse_event maps raw JSON dicts (from claude's stdout) to typed
# Event objects. This is the boundary between the subprocess and our code.
# If claude changes its wire format, THESE tests tell us what broke.


class TestParseAssistantEvent:
    """Parse assistant responses — the most common message type."""

    def test_text_blocks_concatenated(self):
        """Multiple text blocks → single .text string."""
        raw = {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "Hello "},
                    {"type": "text", "text": "world!"},
                ]
            },
        }
        event = Claude._parse_event(raw)
        assert isinstance(event, AssistantEvent)
        assert event.text == "Hello world!"

    def test_tool_use_in_content(self):
        """Tool use blocks in content but excluded from .text."""
        raw = {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "Let me check. "},
                    {"type": "tool_use", "id": "tool_01", "name": "Bash",
                     "input": {"command": "ls"}},
                    {"type": "text", "text": "Done."},
                ]
            },
        }
        event = Claude._parse_event(raw)
        assert event.text == "Let me check. Done."
        assert len(event.content) == 3

    def test_malformed_missing_message(self):
        """Missing 'message' key doesn't crash."""
        event = Claude._parse_event({"type": "assistant"})
        assert isinstance(event, AssistantEvent)
        assert event.text == ""
        assert event.content == []


class TestParseResultEvent:
    """Parse end-of-turn results — cost, session, turn count."""

    def test_all_fields(self):
        raw = {
            "type": "result",
            "session_id": "sess_abc123",
            "total_cost_usd": 0.0042,
            "num_turns": 1,
            "duration_ms": 1500,
            "is_error": False,
        }
        event = Claude._parse_event(raw)
        assert isinstance(event, ResultEvent)
        assert event.session_id == "sess_abc123"
        assert event.cost_usd == 0.0042
        assert event.num_turns == 1
        assert event.is_error is False

    def test_defaults_on_missing_fields(self):
        """Missing fields get safe defaults, not crashes."""
        event = Claude._parse_event({"type": "result"})
        assert isinstance(event, ResultEvent)
        assert event.session_id == ""
        assert event.cost_usd == 0.0
        assert event.is_error is False


class TestParseControlMessages:
    """Parse control messages — permission requests and init responses."""

    def test_permission_request(self):
        raw = {
            "type": "control_request",
            "request_id": "req_123",
            "request": {"subtype": "tool_permission", "tool_name": "Bash"},
        }
        event = Claude._parse_event(raw)
        assert isinstance(event, _ControlRequestEvent)
        assert event.request_id == "req_123"
        assert event.tool_name == "Bash"

    def test_permission_request_falls_back_to_subtype(self):
        """When tool_name is missing, falls back to subtype."""
        raw = {
            "type": "control_request",
            "request_id": "req_456",
            "request": {"subtype": "some_permission"},
        }
        event = Claude._parse_event(raw)
        assert event.tool_name == "some_permission"

    def test_init_response(self):
        raw = {
            "type": "control_response",
            "response": {
                "model": "claude-sonnet-4-20250514",
                "tools": [{"name": "Bash"}, {"name": "Read"}],
                "mcpServers": [{"name": "cortex"}],
            },
        }
        event = Claude._parse_event(raw)
        assert isinstance(event, InitEvent)
        assert event.model == "claude-sonnet-4-20250514"
        assert len(event.tools) == 2
        assert len(event.mcp_servers) == 1


class TestParseStreamEvent:
    """Parse streaming deltas — text, thinking, block boundaries."""

    def test_text_delta(self):
        raw = {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": "Hello"},
            },
        }
        event = Claude._parse_event(raw)
        assert isinstance(event, StreamEvent)
        assert event.delta_text == "Hello"
        assert event.index == 1

    def test_thinking_delta(self):
        raw = {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "Let me think..."},
            },
        }
        event = Claude._parse_event(raw)
        assert event.delta_text == "Let me think..."

    def test_signature_delta_has_no_text(self):
        """Signature deltas have neither text nor thinking."""
        event = StreamEvent(raw={}, inner={
            "delta": {"type": "signature_delta", "signature": "EoU..."},
        })
        assert event.delta_text == ""

    def test_malformed_missing_event(self):
        """Missing 'event' key doesn't crash."""
        event = Claude._parse_event({"type": "stream_event"})
        assert isinstance(event, StreamEvent)
        assert event.event_type == ""
        assert event.delta_text == ""


class TestParseEdgeCases:
    """Unknown or malformed messages."""

    def test_unknown_type_returns_base_event(self):
        raw = {"type": "rate_limit_event", "data": "something"}
        event = Claude._parse_event(raw)
        assert type(event) is Event
        assert event.raw is raw

    def test_empty_dict(self):
        event = Claude._parse_event({})
        assert type(event) is Event

    def test_raw_always_preserved(self):
        """Every event type preserves the original raw dict."""
        for raw in [
            {"type": "assistant", "message": {"content": []}},
            {"type": "result"},
            {"type": "system", "subtype": "x"},
            {"type": "stream_event", "event": {}},
            {"type": "unknown"},
        ]:
            event = Claude._parse_event(raw)
            assert event.raw is raw


# -- Message Formatting: Protocol Correctness --------------------------------
#
# These format methods construct the JSON we send to claude's stdin.
# If any of these are wrong, the subprocess ignores us silently.


class TestFormatMessages:
    """Verify message formats match the stream-json protocol."""

    def test_user_message_structure(self):
        blocks = [{"type": "text", "text": "Hello!"}]
        msg = Claude._format_user_message(blocks)
        assert msg["type"] == "user"
        assert msg["message"]["role"] == "user"
        assert msg["message"]["content"] is blocks
        assert msg["session_id"] == ""
        assert msg["parent_tool_use_id"] is None

    def test_init_request_structure(self):
        msg = Claude._format_init_request()
        assert msg["type"] == "control_request"
        assert msg["request"]["subtype"] == "initialize"
        assert msg["request_id"].startswith("req_0_")

    def test_init_request_ids_unique(self):
        ids = {Claude._format_init_request()["request_id"] for _ in range(20)}
        assert len(ids) == 20

    def test_permission_response_structure(self):
        msg = Claude._format_permission_response("req_42")
        assert msg["type"] == "control_response"
        assert msg["response"]["request_id"] == "req_42"
        assert msg["response"]["response"]["approved"] is True


# -- State Machine Guards ----------------------------------------------------
#
# These prevent callers from doing dumb things like sending messages
# before start() or starting twice.


class TestStateGuards:
    """State machine prevents invalid operations."""

    def test_initial_state_is_idle(self):
        claude = Claude()
        assert claude.state == ClaudeState.IDLE

    async def test_cannot_send_before_start(self):
        claude = Claude()
        with pytest.raises(RuntimeError, match="Cannot send"):
            await claude.send([{"type": "text", "text": "hello"}])

    async def test_cannot_read_events_before_start(self):
        claude = Claude()
        with pytest.raises(RuntimeError, match="Cannot read events"):
            async for _ in claude.events():
                pass

    async def test_stop_from_idle_is_fine(self):
        """Stopping without starting is a no-op, not an error."""
        claude = Claude()
        await claude.stop()
        assert claude.state == ClaudeState.STOPPED

    async def test_double_stop_is_fine(self):
        """Stopping twice is a no-op."""
        claude = Claude()
        await claude.stop()
        await claude.stop()
        assert claude.state == ClaudeState.STOPPED

    async def test_cannot_start_after_stop(self):
        """Once stopped, a Claude instance is done. Make a new one."""
        claude = Claude()
        await claude.stop()
        with pytest.raises(RuntimeError, match="Cannot start"):
            await claude.start()
