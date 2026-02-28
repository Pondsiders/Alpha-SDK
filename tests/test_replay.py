"""Tests for replay.py — JSONL session reader."""

import json
from pathlib import Path

import pytest

from alpha_sdk.engine import AssistantEvent, ResultEvent
from alpha_sdk.replay import _read_jsonl, find_session_path, replay_session


# -- Fixtures ---------------------------------------------------------------


@pytest.fixture
def tmp_session(tmp_path) -> Path:
    """Create a minimal JSONL session file."""
    session_id = "test-session-abc123"
    path = tmp_path / f"{session_id}.jsonl"
    records = [
        # Queue operation — should be skipped
        {
            "type": "queue-operation",
            "operation": "dequeue",
            "sessionId": session_id,
        },
        # User message
        {
            "type": "user",
            "sessionId": session_id,
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "Hello!"}],
            },
        },
        # Assistant response
        {
            "type": "assistant",
            "sessionId": session_id,
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hi there!"}],
            },
        },
        # Second user message
        {
            "type": "user",
            "sessionId": session_id,
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "How are you?"}],
            },
        },
        # Second assistant response
        {
            "type": "assistant",
            "sessionId": session_id,
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'm doing well!"},
                    {"type": "text", "text": " Thanks for asking."},
                ],
            },
        },
    ]
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return path


@pytest.fixture
def empty_session(tmp_path) -> Path:
    """Create an empty JSONL session file."""
    path = tmp_path / "empty-session.jsonl"
    path.write_text("")
    return path


@pytest.fixture
def malformed_session(tmp_path) -> Path:
    """Create a JSONL file with some bad lines."""
    session_id = "malformed-session"
    path = tmp_path / f"{session_id}.jsonl"
    lines = [
        json.dumps({"type": "assistant", "sessionId": session_id,
                     "message": {"role": "assistant",
                                 "content": [{"type": "text", "text": "good"}]}}),
        "this is not json",
        "",
        json.dumps({"type": "assistant", "sessionId": session_id,
                     "message": {"role": "assistant",
                                 "content": [{"type": "text", "text": "also good"}]}}),
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


# -- _read_jsonl -------------------------------------------------------------


class TestReadJsonl:
    def test_reads_valid_records(self, tmp_session):
        records = _read_jsonl(tmp_session)
        assert len(records) == 5

    def test_skips_malformed_lines(self, malformed_session):
        records = _read_jsonl(malformed_session)
        assert len(records) == 2

    def test_empty_file(self, empty_session):
        records = _read_jsonl(empty_session)
        assert records == []


# -- replay_session ----------------------------------------------------------


class TestReplaySession:
    @pytest.mark.asyncio
    async def test_yields_assistant_events(self, tmp_session):
        events = []
        async for event in replay_session(tmp_session):
            events.append(event)

        # 2 assistant messages → 2 AssistantEvents + 2 ResultEvents
        assert len(events) == 4

    @pytest.mark.asyncio
    async def test_event_types_alternate(self, tmp_session):
        events = []
        async for event in replay_session(tmp_session):
            events.append(event)

        assert isinstance(events[0], AssistantEvent)
        assert isinstance(events[1], ResultEvent)
        assert isinstance(events[2], AssistantEvent)
        assert isinstance(events[3], ResultEvent)

    @pytest.mark.asyncio
    async def test_all_events_are_replay(self, tmp_session):
        async for event in replay_session(tmp_session):
            assert event.is_replay is True

    @pytest.mark.asyncio
    async def test_content_is_preserved(self, tmp_session):
        events = []
        async for event in replay_session(tmp_session):
            if isinstance(event, AssistantEvent):
                events.append(event)

        # First assistant message
        assert events[0].text == "Hi there!"
        # Second assistant message (two text blocks)
        assert events[1].text == "I'm doing well! Thanks for asking."

    @pytest.mark.asyncio
    async def test_result_events_have_session_id(self, tmp_session):
        results = []
        async for event in replay_session(tmp_session):
            if isinstance(event, ResultEvent):
                results.append(event)

        assert results[0].session_id == "test-session-abc123"
        assert results[1].session_id == "test-session-abc123"

    @pytest.mark.asyncio
    async def test_turn_numbers_increment(self, tmp_session):
        results = []
        async for event in replay_session(tmp_session):
            if isinstance(event, ResultEvent):
                results.append(event)

        assert results[0].num_turns == 1
        assert results[1].num_turns == 2

    @pytest.mark.asyncio
    async def test_empty_session(self, empty_session):
        events = []
        async for event in replay_session(empty_session):
            events.append(event)
        assert events == []

    @pytest.mark.asyncio
    async def test_malformed_lines_skipped(self, malformed_session):
        events = []
        async for event in replay_session(malformed_session):
            events.append(event)

        # 2 valid assistant messages → 2 AssistantEvents + 2 ResultEvents
        assert len(events) == 4

    @pytest.mark.asyncio
    async def test_raw_preserved_on_assistant_events(self, tmp_session):
        async for event in replay_session(tmp_session):
            if isinstance(event, AssistantEvent):
                assert event.raw.get("type") == "assistant"
                break

    @pytest.mark.asyncio
    async def test_user_messages_not_yielded(self, tmp_session):
        """User messages are context, not events — they shouldn't be yielded."""
        async for event in replay_session(tmp_session):
            assert isinstance(event, (AssistantEvent, ResultEvent))


# -- find_session_path -------------------------------------------------------


class TestFindSessionPath:
    def test_finds_existing_session(self, tmp_path):
        session_id = "abc-123-def"
        (tmp_path / f"{session_id}.jsonl").touch()
        result = find_session_path(session_id, search_dirs=[tmp_path])
        assert result is not None
        assert result.name == f"{session_id}.jsonl"

    def test_returns_none_for_missing_session(self, tmp_path):
        result = find_session_path("nonexistent", search_dirs=[tmp_path])
        assert result is None

    def test_searches_multiple_dirs(self, tmp_path):
        dir1 = tmp_path / "project1"
        dir2 = tmp_path / "project2"
        dir1.mkdir()
        dir2.mkdir()

        session_id = "found-in-dir2"
        (dir2 / f"{session_id}.jsonl").touch()

        result = find_session_path(session_id, search_dirs=[dir1, dir2])
        assert result is not None
        assert result.parent == dir2

    def test_empty_search_dirs(self):
        result = find_session_path("anything", search_dirs=[])
        assert result is None
