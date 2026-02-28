"""test_engine.py — Unit tests for the claude subprocess wrapper.

Tests the Engine's static/pure methods: event parsing, message formatting,
state machine transitions, and event type behavior. No subprocess needed.
"""

import pytest

from alpha_sdk.engine import (
    Engine,
    EngineState,
    Event,
    InitEvent,
    AssistantEvent,
    ResultEvent,
    SystemEvent,
    ErrorEvent,
    _ControlRequestEvent,
)
from alpha_sdk.proxy import CompactConfig


# -- Event Parsing -----------------------------------------------------------
# Engine._parse_event maps raw JSON dicts to typed Event objects.
# This is the contract between the wire protocol and our type system.


class TestParseAssistantEvent:
    """Parse assistant messages with content blocks."""

    def test_single_text_block(self):
        raw = {
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": "Hello, world!"}]
            },
        }
        event = Engine._parse_event(raw)
        assert isinstance(event, AssistantEvent)
        assert event.text == "Hello, world!"
        assert event.raw is raw

    def test_multiple_text_blocks(self):
        raw = {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "Hello "},
                    {"type": "text", "text": "world!"},
                ]
            },
        }
        event = Engine._parse_event(raw)
        assert event.text == "Hello world!"

    def test_mixed_content_blocks(self):
        """Tool use blocks appear in content but not in .text."""
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
        event = Engine._parse_event(raw)
        assert isinstance(event, AssistantEvent)
        assert event.text == "Let me check. Done."
        assert len(event.content) == 3

    def test_empty_content(self):
        raw = {"type": "assistant", "message": {"content": []}}
        event = Engine._parse_event(raw)
        assert isinstance(event, AssistantEvent)
        assert event.text == ""

    def test_missing_message_key(self):
        """Malformed assistant message — missing 'message' key."""
        raw = {"type": "assistant"}
        event = Engine._parse_event(raw)
        assert isinstance(event, AssistantEvent)
        assert event.content == []
        assert event.text == ""


class TestParseResultEvent:
    """Parse end-of-turn result messages."""

    def test_successful_result(self):
        raw = {
            "type": "result",
            "session_id": "sess_abc123",
            "total_cost_usd": 0.0042,
            "num_turns": 1,
            "duration_ms": 1500,
            "is_error": False,
        }
        event = Engine._parse_event(raw)
        assert isinstance(event, ResultEvent)
        assert event.session_id == "sess_abc123"
        assert event.cost_usd == 0.0042
        assert event.num_turns == 1
        assert event.duration_ms == 1500
        assert event.is_error is False

    def test_error_result(self):
        raw = {"type": "result", "is_error": True, "session_id": ""}
        event = Engine._parse_event(raw)
        assert isinstance(event, ResultEvent)
        assert event.is_error is True

    def test_missing_fields_default(self):
        """Missing fields should use sensible defaults."""
        raw = {"type": "result"}
        event = Engine._parse_event(raw)
        assert isinstance(event, ResultEvent)
        assert event.session_id == ""
        assert event.cost_usd == 0.0
        assert event.num_turns == 0
        assert event.is_error is False


class TestParseSystemEvent:
    """Parse system messages."""

    def test_system_with_subtype(self):
        raw = {"type": "system", "subtype": "session_id"}
        event = Engine._parse_event(raw)
        assert isinstance(event, SystemEvent)
        assert event.subtype == "session_id"

    def test_system_without_subtype(self):
        raw = {"type": "system"}
        event = Engine._parse_event(raw)
        assert isinstance(event, SystemEvent)
        assert event.subtype == ""


class TestParseControlRequest:
    """Parse permission requests (handled internally, not yielded)."""

    def test_tool_permission(self):
        raw = {
            "type": "control_request",
            "request_id": "req_123",
            "request": {
                "subtype": "tool_permission",
                "tool_name": "Bash",
            },
        }
        event = Engine._parse_event(raw)
        assert isinstance(event, _ControlRequestEvent)
        assert event.request_id == "req_123"
        assert event.tool_name == "Bash"

    def test_missing_tool_name_falls_back_to_subtype(self):
        raw = {
            "type": "control_request",
            "request_id": "req_456",
            "request": {"subtype": "some_permission"},
        }
        event = Engine._parse_event(raw)
        assert isinstance(event, _ControlRequestEvent)
        assert event.tool_name == "some_permission"


class TestParseControlResponse:
    """Parse init handshake response as InitEvent."""

    def test_init_with_capabilities(self):
        raw = {
            "type": "control_response",
            "response": {
                "subtype": "init_success",
                "model": "claude-sonnet-4-20250514",
                "tools": [
                    {"name": "Bash", "type": "tool"},
                    {"name": "Read", "type": "tool"},
                ],
                "mcpServers": [{"name": "cortex"}],
            },
        }
        event = Engine._parse_event(raw)
        assert isinstance(event, InitEvent)
        assert event.model == "claude-sonnet-4-20250514"
        assert len(event.tools) == 2
        assert len(event.mcp_servers) == 1

    def test_init_empty_capabilities(self):
        raw = {"type": "control_response", "response": {}}
        event = Engine._parse_event(raw)
        assert isinstance(event, InitEvent)
        assert event.model == ""
        assert event.tools == []
        assert event.mcp_servers == []


class TestParseEdgeCases:
    """Edge cases in event parsing."""

    def test_unknown_type(self):
        raw = {"type": "stream_event", "data": "something"}
        event = Engine._parse_event(raw)
        assert type(event) is Event
        assert event.raw is raw

    def test_empty_dict(self):
        raw = {}
        event = Engine._parse_event(raw)
        assert type(event) is Event

    def test_no_type_key(self):
        raw = {"foo": "bar"}
        event = Engine._parse_event(raw)
        assert type(event) is Event


# -- Message Formatting -------------------------------------------------------
# Static methods that construct protocol-correct JSON messages.


class TestFormatUserMessage:
    """Test user message construction."""

    def test_basic_message(self):
        msg = Engine._format_user_message("Hello!")
        assert msg["type"] == "user"
        assert msg["message"]["role"] == "user"
        assert msg["message"]["content"] == "Hello!"
        assert msg["session_id"] == ""
        assert msg["parent_tool_use_id"] is None

    def test_preserves_content_exactly(self):
        text = "line 1\nline 2\n  indented\ttabbed"
        msg = Engine._format_user_message(text)
        assert msg["message"]["content"] == text

    def test_empty_string(self):
        msg = Engine._format_user_message("")
        assert msg["message"]["content"] == ""


class TestFormatInitRequest:
    """Test init handshake request construction."""

    def test_structure(self):
        msg = Engine._format_init_request()
        assert msg["type"] == "control_request"
        assert msg["request"]["subtype"] == "initialize"
        assert msg["request"]["hooks"] == {}
        assert msg["request"]["agents"] == {}

    def test_request_id_format(self):
        msg = Engine._format_init_request()
        assert msg["request_id"].startswith("req_0_")
        assert len(msg["request_id"]) == 14  # "req_0_" + 8 hex chars

    def test_unique_ids(self):
        """Each request should have a unique ID."""
        ids = {Engine._format_init_request()["request_id"] for _ in range(20)}
        assert len(ids) == 20


class TestFormatPermissionResponse:
    """Test auto-approve response construction."""

    def test_structure(self):
        msg = Engine._format_permission_response("req_42")
        assert msg["type"] == "control_response"
        assert msg["response"]["subtype"] == "success"
        assert msg["response"]["request_id"] == "req_42"
        assert msg["response"]["response"]["approved"] is True


# -- State Machine ------------------------------------------------------------


class TestEngineState:
    """Test lifecycle state transitions."""

    def test_initial_state(self):
        engine = Engine()
        assert engine.state == EngineState.IDLE

    def test_no_pid_before_start(self):
        engine = Engine()
        assert engine.pid is None

    async def test_cannot_send_before_start(self):
        engine = Engine()
        with pytest.raises(RuntimeError, match="Cannot send"):
            await engine.send("hello")

    async def test_cannot_read_events_before_start(self):
        engine = Engine()
        with pytest.raises(RuntimeError, match="Cannot read events"):
            async for _ in engine.events():
                pass

    async def test_cannot_start_twice(self):
        """Starting a non-IDLE engine should fail."""
        engine = Engine()
        await engine.stop()  # IDLE → STOPPED
        with pytest.raises(RuntimeError, match="Cannot start"):
            await engine.start()

    async def test_stop_when_idle(self):
        """Stopping an engine that was never started is fine."""
        engine = Engine()
        await engine.stop()
        assert engine.state == EngineState.STOPPED

    async def test_double_stop(self):
        """Stopping an already-stopped engine is a no-op."""
        engine = Engine()
        await engine.stop()
        await engine.stop()
        assert engine.state == EngineState.STOPPED


# -- Engine Configuration ----------------------------------------------------


class TestEngineConfig:
    """Test that configuration is stored correctly."""

    def test_defaults(self):
        engine = Engine()
        assert engine.model == "claude-sonnet-4-20250514"
        assert engine.system_prompt is None
        assert engine.mcp_config is None
        assert engine.permission_mode == "bypassPermissions"

    def test_custom_config(self):
        engine = Engine(
            model="claude-opus-4-20250514",
            system_prompt="You are a duck.",
            mcp_config="/path/to/config.json",
            permission_mode="default",
        )
        assert engine.model == "claude-opus-4-20250514"
        assert engine.system_prompt == "You are a duck."
        assert engine.mcp_config == "/path/to/config.json"
        assert engine.permission_mode == "default"

    def test_extra_args_default_empty(self):
        engine = Engine()
        assert engine.extra_args == []

    def test_extra_args_stored(self):
        args = ["--tools", "", "--disable-slash-commands", "--no-chrome"]
        engine = Engine(extra_args=args)
        assert engine.extra_args == args

    def test_compact_config_default_none(self):
        engine = Engine()
        assert engine.compact_config is None

    def test_compact_config_stored(self):
        config = CompactConfig(system="s", prompt="p", continuation="c")
        engine = Engine(compact_config=config)
        assert engine.compact_config is config


# -- Proxy property defaults (before start) -----------------------------------


class TestProxyPropertyDefaults:
    """Proxy properties return safe defaults when no proxy is running."""

    def test_token_count_default(self):
        engine = Engine()
        assert engine.token_count == 0

    def test_context_window_default(self):
        engine = Engine()
        assert engine.context_window == 200_000

    def test_usage_7d_default(self):
        engine = Engine()
        assert engine.usage_7d is None

    def test_usage_5h_default(self):
        engine = Engine()
        assert engine.usage_5h is None

    def test_reset_token_count_no_proxy(self):
        """reset_token_count is a no-op when no proxy exists."""
        engine = Engine()
        engine.reset_token_count()  # Should not raise


# -- AssistantEvent text property ---------------------------------------------


class TestAssistantEventText:
    """Test the .text convenience property."""

    def test_single_block(self):
        event = AssistantEvent(
            raw={},
            content=[{"type": "text", "text": "hello"}],
        )
        assert event.text == "hello"

    def test_concatenation(self):
        event = AssistantEvent(
            raw={},
            content=[
                {"type": "text", "text": "hello "},
                {"type": "text", "text": "world"},
            ],
        )
        assert event.text == "hello world"

    def test_skips_non_text(self):
        event = AssistantEvent(
            raw={},
            content=[
                {"type": "text", "text": "before "},
                {"type": "tool_use", "id": "x", "name": "Bash"},
                {"type": "text", "text": "after"},
            ],
        )
        assert event.text == "before after"

    def test_empty(self):
        event = AssistantEvent(raw={}, content=[])
        assert event.text == ""

    def test_missing_text_key(self):
        """Block has type=text but no text key — should not crash."""
        event = AssistantEvent(
            raw={},
            content=[{"type": "text"}],
        )
        assert event.text == ""
