"""Preprocessing pipeline helpers for AlphaClient.

Pure functions for formatting, approach lights, orientation assembly,
and text extraction. These are the building blocks that _build_user_content()
orchestrates on each turn.
"""

from typing import Any

import pendulum

from .system_prompt import assemble


def extract_prompt_text(prompt: str | list[dict[str, Any]]) -> str:
    """Extract plain text from a prompt (string or content blocks).

    Used by send() for span naming and by _build_user_content() for
    memory operations.
    """
    if isinstance(prompt, str):
        return prompt
    text_parts = [b.get("text", "") for b in prompt if b.get("type") == "text"]
    return " ".join(text_parts)


def relative_time(created_at: str) -> str:
    """Format a created_at timestamp as human-readable relative time."""
    try:
        dt = pendulum.parse(created_at).in_tz("America/Los_Angeles")
        now = pendulum.now("America/Los_Angeles")
        diff = now.diff(dt)
        if diff.in_days() == 0:
            return f"today at {dt.format('h:mm A')}"
        elif diff.in_days() == 1:
            return f"yesterday at {dt.format('h:mm A')}"
        elif diff.in_days() < 7:
            return f"{diff.in_days()} days ago"
        elif diff.in_days() < 30:
            weeks = diff.in_days() // 7
            return f"{weeks} week{'s' if weeks > 1 else ''} ago"
        else:
            return dt.format("ddd MMM D YYYY")
    except Exception:
        return created_at  # fallback to raw string


def format_memory(memory: dict) -> str:
    """Format a memory for inclusion in user content.

    Creates human-readable memory text with relative timestamps.
    """
    mem_id = memory.get("id", "?")
    content = memory.get("content", "").strip()
    score = memory.get("score")
    relative = relative_time(memory.get("created_at", ""))

    # Include score if present (helps with debugging/transparency)
    score_str = f", score {score:.2f}" if score is not None else ""
    return f"## Memory #{mem_id} ({relative}{score_str})\n{content}"


def format_image_recall(results: list[dict[str, Any]]) -> str:
    """Format memories for image-triggered recall.

    Text-only breadcrumbs — no binary image injection.
    This prevents recursion: images reminding of images that have images.
    """
    lines = ["🔍 This image reminds me of:"]
    for item in results:
        mem_id = item.get("id", "?")
        metadata = item.get("metadata", {})
        relative = relative_time(metadata.get("created_at", ""))
        content = item.get("content", "").strip()

        # First line only, truncated for breadcrumb
        first_line = content.split("\n")[0]
        if len(first_line) > 120:
            first_line = first_line[:117] + "..."

        # Flag memories that have attached images
        image_flag = " [📷 attached]" if metadata.get("image_path") else ""

        lines.append(f"• Memory #{mem_id} ({relative}): {first_line}{image_flag}")

    return "\n".join(lines)


def get_approach_light(
    token_count: int,
    context_window: int,
    warned_level: int,
) -> tuple[str | None, int]:
    """Check context usage and return an approach light warning if threshold crossed.

    Two tiers:
    - Amber (65%): gentle heads-up to start thinking about pausing
    - Red (75%): stern warning to wrap up or hand off

    Only fires once per tier (resets after compaction).

    Args:
        token_count: Current token usage
        context_window: Maximum context window size
        warned_level: Current warning level (0=none, 1=amber, 2=red)

    Returns:
        (warning_text, new_warned_level) — text is None if no warning needed.
    """
    if context_window == 0 or token_count == 0:
        return None, warned_level

    pct = token_count / context_window

    if pct >= 0.75 and warned_level < 2:
        return (
            f"## ⚠️ Context Warning — RED ({pct:.0%})\n\n"
            f"You're at {token_count:,} of {context_window:,} tokens. "
            "This is not a drill. Wrap up what you're doing and either hand off "
            "(use the hand-off tool with instructions for next-you) or let Jeffery know "
            "it's time to compact. Starting new topics now is risky — you may lose context "
            "before you can finish them."
        ), 2

    if pct >= 0.65 and warned_level < 1:
        return (
            f"## 🟡 Context Warning — AMBER ({pct:.0%})\n\n"
            f"You're at {token_count:,} of {context_window:,} tokens. "
            "Start thinking about a good place to pause. You don't need to stop immediately, "
            "but keep an eye on where the conversation is heading. If there's something important "
            "to store or hand off, now's a good time to start thinking about it."
        ), 1

    return None, warned_level


async def build_orientation(
    client_name: str,
    hostname: str | None = None,
) -> list[dict[str, Any]]:
    """Build orientation blocks for session start.

    This includes everything except the soul (which is in system prompt):
    - Capsules (yesterday, last night)
    - Letter from last night
    - Today so far
    - Here (client, machine, weather)
    - ALPHA.md context files
    - Events
    - Todos
    """
    all_blocks = await assemble(
        client=client_name,
        hostname=hostname,
    )

    # Skip the first block (which is the soul)
    # The soul starts with "# Alpha\n\n"
    orientation_blocks = []
    for block in all_blocks:
        text = block.get("text", "")
        if not text.startswith("# Alpha\n\n"):
            orientation_blocks.append(block)

    return orientation_blocks
