"""Archive conversation turns to Postgres (Scribe).

Records user and assistant messages to scribe.messages after each turn.
This is the safety net—if context closes unexpectedly, we have the transcript.

Messages are inserted with timestamps, roles, and session IDs.
Embeddings are generated asynchronously after insert (fire-and-forget).
"""

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import logfire

from .memories.db import get_pool
from .memories.embeddings import embed_document

# Configuration
ARCHIVE_ENABLED = os.environ.get("ALPHA_ARCHIVE", "1").lower() in ("1", "true", "yes")


@dataclass
class ArchiveResult:
    """Result of an archive operation."""

    success: bool
    error: str | None = None
    rows_inserted: int = 0
    row_ids: list[int] = field(default_factory=list)


def _extract_text_content(content: str | list[Any]) -> str:
    """Extract text from content (which may be a string or list of content blocks)."""
    if isinstance(content, str):
        return content

    # List of content blocks - extract text parts
    texts = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                texts.append(block.get("text", ""))
    return "\n".join(texts)


async def _embed_and_update(row_id: int, content: str) -> bool:
    """Embed content and update the row. Returns success status."""
    try:
        embedding = await embed_document(content)

        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE scribe.messages
                SET embedding = $1::vector
                WHERE id = $2
                """,
                str(embedding),
                row_id,
            )

        return True

    except Exception as e:
        logfire.error("Archive embed failed for row {row_id}: {error}", row_id=row_id, error=str(e))
        return False


async def _embed_rows(row_ids: list[int], contents: list[str]) -> None:
    """Embed multiple rows in parallel. Fire-and-forget."""
    if not row_ids:
        return

    tasks = [
        _embed_and_update(row_id, content)
        for row_id, content in zip(row_ids, contents)
    ]
    await asyncio.gather(*tasks, return_exceptions=True)


async def archive_turn(
    user_content: str | list[Any],
    assistant_content: str,
    session_id: str | None,
    timestamp: datetime | None = None,
) -> ArchiveResult:
    """Archive a conversation turn to scribe.messages.

    Inserts one row for the user message and one for the assistant response.
    Uses ON CONFLICT to avoid duplicates (keyed on timestamp + role + content hash).
    After successful insert, fires off async embedding (non-blocking).

    Args:
        user_content: The user's message (string or list of content blocks)
        assistant_content: The assistant's full response text
        session_id: Current session ID (can be None for new sessions)
        timestamp: When the turn occurred (defaults to now)

    Returns:
        ArchiveResult with success status, row IDs, and any error message
    """
    if not ARCHIVE_ENABLED:
        return ArchiveResult(success=True, rows_inserted=0)

    with logfire.span("archive", session_id=session_id):
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Extract text from potentially complex content
        user_text = _extract_text_content(user_content)

        if not user_text.strip() and not assistant_content.strip():
            return ArchiveResult(success=True, rows_inserted=0)  # Nothing to archive

        try:
            pool = await get_pool()

            # Build rows to insert
            rows: list[tuple[datetime, str, str, str | None]] = []
            contents: list[str] = []  # For embedding later

            if user_text.strip():
                rows.append((timestamp, "human", user_text, session_id))
                contents.append(user_text)

            if assistant_content.strip():
                # Slight offset for assistant timestamp to maintain ordering
                assistant_ts = timestamp.replace(microsecond=min(timestamp.microsecond + 1, 999999))
                rows.append((assistant_ts, "assistant", assistant_content, session_id))
                contents.append(assistant_content)

            if not rows:
                return ArchiveResult(success=True, rows_inserted=0)

            # Insert into Postgres, returning IDs
            row_ids: list[int] = []
            inserted_contents: list[str] = []
            async with pool.acquire() as conn:
                for i, row in enumerate(rows):
                    result = await conn.fetchval(
                        """
                        INSERT INTO scribe.messages (timestamp, role, content, session_id)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (timestamp, role, md5(content)) DO NOTHING
                        RETURNING id
                        """,
                        row[0], row[1], row[2], row[3],
                    )
                    if result:
                        row_ids.append(result)
                        inserted_contents.append(contents[i])

            # Fire off embedding task (non-blocking)
            if row_ids:
                asyncio.create_task(_embed_rows(row_ids, inserted_contents))

            logfire.info(
                "Archive: {rows} rows inserted, {embeds} embeddings queued",
                rows=len(row_ids),
                embeds=len(row_ids),
            )

            return ArchiveResult(
                success=True,
                rows_inserted=len(row_ids),
                row_ids=row_ids,
            )

        except Exception as e:
            logfire.error("Archive failed: {error}", error=str(e))
            return ArchiveResult(success=False, error=f"Archive failed: {e}")
