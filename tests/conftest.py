"""Shared test fixtures for Alpha SDK Next."""

import json

import pytest


# -- Markers ------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests that need live claude process or database",
    )


# -- Canned protocol fixtures ------------------------------------------------
# Based on quack-wire.py observations (Feb 26, 2026).
# These represent the JSON messages claude sends/receives over stdio.


@pytest.fixture
def sample_init_response():
    """The capabilities advertisement claude emits on startup."""
    return {
        "type": "system",
        "subtype": "init",
        "apiKeySource": "MAX_SUBSCRIPTION",
        "claudeAiVersion": "2025.1",
        "configDir": "/home/user/.claude",
        "cwd": "/tmp",
        "model": "claude-sonnet-4-20250514",
        "permissionMode": "default",
        "tools": [
            {"name": "Bash", "type": "tool"},
            {"name": "Read", "type": "tool"},
            {"name": "Write", "type": "tool"},
            {"name": "Edit", "type": "tool"},
            {"name": "Glob", "type": "tool"},
            {"name": "Grep", "type": "tool"},
        ],
        "mcpServers": [],
    }


@pytest.fixture
def sample_text_event():
    """A text content block from claude's response."""
    return {"type": "assistant", "subtype": "text", "text": "Hello! How can I help?"}


@pytest.fixture
def sample_tool_use_event():
    """A tool use event — claude wants to call a tool."""
    return {
        "type": "assistant",
        "subtype": "tool_use",
        "name": "Bash",
        "id": "tool_01ABC",
        "input": {"command": "echo hello"},
    }


@pytest.fixture
def sample_tool_result_event():
    """A tool result — claude reporting what a tool returned."""
    return {
        "type": "assistant",
        "subtype": "tool_result",
        "name": "Bash",
        "id": "tool_01ABC",
        "output": "hello\n",
    }


@pytest.fixture
def sample_result_event():
    """End-of-turn result event."""
    return {
        "type": "result",
        "subtype": "success",
        "duration_ms": 1234,
        "duration_api_ms": 987,
        "is_error": False,
        "num_turns": 1,
        "session_id": "abc123-def456",
        "total_cost_usd": 0.0,
    }


@pytest.fixture
def sample_user_message():
    """A user message in the format we send to claude's stdin."""
    return {
        "type": "user",
        "message": {
            "role": "user",
            "content": "Hello, world!",
        },
    }


@pytest.fixture
def sample_conversation_turn(sample_text_event, sample_result_event):
    """A complete conversation turn: text response followed by result."""
    return [sample_text_event, sample_result_event]


# -- Helpers ------------------------------------------------------------------


@pytest.fixture
def ndjson_lines():
    """Helper: convert a list of dicts to newline-delimited JSON bytes."""
    def _make(events: list[dict]) -> bytes:
        lines = [json.dumps(e) for e in events]
        return ("\n".join(lines) + "\n").encode()
    return _make
