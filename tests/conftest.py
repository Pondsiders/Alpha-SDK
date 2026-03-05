"""Shared test fixtures for Alpha SDK.

Protocol fixtures based on the stream-json wire format claude uses
over stdio. These represent actual JSON messages from observation.
"""

import json

import pytest


# -- Markers ------------------------------------------------------------------


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests that spawn a live claude process (cost: ~$0.001/test)",
    )


# -- Canned protocol fixtures ------------------------------------------------


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
def sample_result():
    """End-of-turn result message."""
    return {
        "type": "result",
        "session_id": "sess_abc123def456",
        "total_cost_usd": 0.0042,
        "num_turns": 1,
        "duration_ms": 1234,
        "is_error": False,
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


# -- Helpers ------------------------------------------------------------------


@pytest.fixture
def ndjson_lines():
    """Helper: convert a list of dicts to newline-delimited JSON bytes."""
    def _make(events: list[dict]) -> bytes:
        lines = [json.dumps(e) for e in events]
        return ("\n".join(lines) + "\n").encode()
    return _make
