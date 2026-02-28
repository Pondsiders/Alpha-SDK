"""Shared test fixtures for Alpha SDK Next.

Protocol fixtures based on quack-raw.py and quack-wire.py observations
(Feb 26, 2026). These represent the actual JSON messages claude
sends/receives over stdio.
"""

import json

import pytest


# -- Markers ------------------------------------------------------------------


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests that need live claude process or database",
    )


# -- Canned protocol fixtures ------------------------------------------------


@pytest.fixture
def sample_init_request():
    """The init handshake request we send to claude."""
    return {
        "type": "control_request",
        "request_id": "req_0_deadbeef",
        "request": {
            "subtype": "initialize",
            "hooks": {},
            "agents": {},
        },
    }


@pytest.fixture
def sample_init_response():
    """The control_response claude sends back after init."""
    return {
        "type": "control_response",
        "response": {
            "subtype": "init_success",
            "model": "claude-sonnet-4-20250514",
            "tools": [
                {"name": "Bash", "type": "tool"},
                {"name": "Read", "type": "tool"},
                {"name": "Write", "type": "tool"},
                {"name": "Edit", "type": "tool"},
                {"name": "Glob", "type": "tool"},
                {"name": "Grep", "type": "tool"},
            ],
            "mcpServers": [],
        },
    }


@pytest.fixture
def sample_assistant_message():
    """An assistant response with a text content block."""
    return {
        "type": "assistant",
        "message": {
            "content": [{"type": "text", "text": "Hello! How can I help?"}]
        },
    }


@pytest.fixture
def sample_tool_use_message():
    """An assistant response containing a tool use block."""
    return {
        "type": "assistant",
        "message": {
            "content": [
                {"type": "text", "text": "Let me check that."},
                {
                    "type": "tool_use",
                    "id": "tool_01ABC",
                    "name": "Bash",
                    "input": {"command": "echo hello"},
                },
            ]
        },
    }


@pytest.fixture
def sample_control_request():
    """A permission request from claude (e.g., tool approval)."""
    return {
        "type": "control_request",
        "request_id": "req_42_cafebabe",
        "request": {
            "subtype": "tool_permission",
            "tool_name": "Bash",
        },
    }


@pytest.fixture
def sample_result():
    """End-of-turn result message."""
    return {
        "type": "result",
        "session_id": "sess_abc123def456",
        "total_cost_usd": 0.0042,
        "num_turns": 1,
        "duration_ms": 1234,
        "duration_api_ms": 987,
        "is_error": False,
        "usage": {
            "input_tokens": 150,
            "output_tokens": 42,
        },
    }


@pytest.fixture
def sample_system_message():
    """A system message from claude."""
    return {
        "type": "system",
        "subtype": "session_id",
        "session_id": "sess_abc123def456",
    }


@pytest.fixture
def sample_user_message():
    """A user message in the format we send to claude's stdin."""
    return {
        "type": "user",
        "session_id": "",
        "message": {"role": "user", "content": [{"type": "text", "text": "Hello, world!"}]},
        "parent_tool_use_id": None,
    }


@pytest.fixture
def sample_conversation_turn(sample_assistant_message, sample_result):
    """A minimal conversation turn: assistant text + result."""
    return [sample_assistant_message, sample_result]


# -- Helpers ------------------------------------------------------------------


@pytest.fixture
def ndjson_lines():
    """Helper: convert a list of dicts to newline-delimited JSON bytes."""
    def _make(events: list[dict]) -> bytes:
        lines = [json.dumps(e) for e in events]
        return ("\n".join(lines) + "\n").encode()
    return _make
