"""Smoke tests — verify test infrastructure works."""

import json

from alpha_sdk import __version__


def test_version():
    """Package version is 2.0.0 alpha."""
    assert __version__ == "2.0.0a0"


def test_fixtures_load(sample_init_response, sample_assistant_message, sample_result):
    """Conftest fixtures are accessible and well-formed."""
    assert sample_init_response["type"] == "control_response"
    assert "tools" in sample_init_response["response"]

    assert sample_assistant_message["type"] == "assistant"
    assert "message" in sample_assistant_message
    assert sample_assistant_message["message"]["content"][0]["type"] == "text"

    assert sample_result["type"] == "result"
    assert "session_id" in sample_result


def test_ndjson_helper(ndjson_lines, sample_assistant_message, sample_result):
    """The ndjson_lines helper produces valid newline-delimited JSON."""
    events = [sample_assistant_message, sample_result]
    raw = ndjson_lines(events)

    assert isinstance(raw, bytes)
    lines = raw.decode().strip().split("\n")
    assert len(lines) == 2

    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["type"] == "assistant"
    assert parsed[1]["type"] == "result"


def test_user_message_format(sample_user_message):
    """User messages have the expected wire protocol structure."""
    assert sample_user_message["type"] == "user"
    assert sample_user_message["message"]["role"] == "user"
    assert isinstance(sample_user_message["message"]["content"], str)
    assert sample_user_message["session_id"] == ""
    assert sample_user_message["parent_tool_use_id"] is None
