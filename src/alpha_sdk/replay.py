"""replay.py — Read claude JSONL session files and yield typed events.

When resuming a session, we need to replay the conversation history so
consumers can render it (draw the chat, rebuild state, etc.). This module
reads claude's JSONL session transcript and yields the same Event types
the engine produces — but with is_replay=True on every event.

The consumer's existing event loop just works. Events are events — some
came from disk (fast), some from the subprocess (real-time). The is_replay
flag lets consumers distinguish if they want to (batch-render history,
skip animation) but they don't have to.

JSONL format (observed from claude's session files):
  - type: "user" + message.role: "user"    → user input (not replayed as events)
  - type: "assistant" + message             → assistant response → AssistantEvent
  - type: "queue-operation"                 → internal bookkeeping, skipped

Each assistant message in the transcript becomes an AssistantEvent followed
by a synthetic ResultEvent (marking end of that turn). This matches the
live event stream's shape: content, then result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import AsyncIterator

from .engine import AssistantEvent, ResultEvent


async def replay_session(path: Path) -> AsyncIterator[AssistantEvent | ResultEvent]:
    """Read a JSONL session file and yield replay events.

    Yields AssistantEvent + ResultEvent pairs for each assistant turn
    in the transcript. All events have is_replay=True.

    Args:
        path: Path to the JSONL session file.

    Yields:
        AssistantEvent for each assistant message, followed by
        ResultEvent marking end of that turn.
    """
    session_id = path.stem  # UUID is the filename

    turn_number = 0
    for record in _read_jsonl(path):
        record_type = record.get("type", "")

        if record_type == "assistant":
            message = record.get("message", {})
            content = message.get("content", [])

            yield AssistantEvent(
                raw=record,
                content=content,
                is_replay=True,
            )

            turn_number += 1
            yield ResultEvent(
                raw={},
                session_id=session_id,
                num_turns=turn_number,
                is_replay=True,
            )


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file and return parsed records.

    Skips blank lines and malformed JSON silently (matching engine behavior).
    """
    records = []
    with open(path) as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                records.append(json.loads(text))
            except json.JSONDecodeError:
                continue
    return records


def find_session_path(session_id: str, search_dirs: list[Path] | None = None) -> Path | None:
    """Find the JSONL file for a session ID.

    Searches claude's project directories for a matching session file.

    Args:
        session_id: The UUID of the session.
        search_dirs: Directories to search. Defaults to ~/.claude/projects/
                     subdirectories.

    Returns:
        Path to the JSONL file, or None if not found.
    """
    if search_dirs is None:
        claude_dir = Path.home() / ".claude" / "projects"
        if claude_dir.exists():
            search_dirs = [d for d in claude_dir.iterdir() if d.is_dir()]
        else:
            return None

    for directory in search_dirs:
        candidate = directory / f"{session_id}.jsonl"
        if candidate.exists():
            return candidate

    return None
