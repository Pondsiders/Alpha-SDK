"""Smoke tests — verify test infrastructure works."""

import json

from alpha_sdk import __version__


def test_version():
    """Package version is 2.0.0 alpha."""
    assert __version__ == "2.0.0a0"


def test_fixtures_load(sample_init_response, sample_text_event, sample_result_event):
    """Conftest fixtures are accessible and well-formed."""
    assert sample_init_response["type"] == "system"
    assert sample_init_response["subtype"] == "init"
    assert isinstance(sample_init_response["tools"], list)

    assert sample_text_event["type"] == "assistant"
    assert "text" in sample_text_event

    assert sample_result_event["type"] == "result"
    assert "session_id" in sample_result_event


def test_ndjson_helper(ndjson_lines, sample_text_event, sample_result_event):
    """The ndjson_lines helper produces valid newline-delimited JSON."""
    events = [sample_text_event, sample_result_event]
    raw = ndjson_lines(events)

    assert isinstance(raw, bytes)
    lines = raw.decode().strip().split("\n")
    assert len(lines) == 2

    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["type"] == "assistant"
    assert parsed[1]["type"] == "result"


def test_user_message_format(sample_user_message):
    """User messages have the expected structure."""
    assert sample_user_message["type"] == "user"
    assert sample_user_message["message"]["role"] == "user"
    assert isinstance(sample_user_message["message"]["content"], str)
