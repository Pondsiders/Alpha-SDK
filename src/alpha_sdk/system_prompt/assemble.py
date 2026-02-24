"""System prompt assembly - weaving all threads together.

This is the main entry point for building Alpha's complete system prompt.
Each piece is fetched and assembled into a coherent whole.
"""

import asyncio
import os

import pendulum
import redis.asyncio as aioredis

from .soul import get_soul
from .capsules import get_capsules
from .here import get_here
from .context import load_context
from .calendar import get_events
from .todos import get_todos

REDIS_URL = os.environ.get("REDIS_URL", "redis://alpha-pi:6379")


async def _get_redis() -> aioredis.Redis:
    """Get async Redis connection."""
    return aioredis.from_url(REDIS_URL, decode_responses=True)


async def _get_hud_extras() -> dict:
    """Fetch extra HUD data: to_self letter and today_so_far summary."""
    try:
        r = await _get_redis()
        to_self, to_self_time, today_so_far, today_so_far_time = await asyncio.gather(
            r.get("systemprompt:past:to_self"),
            r.get("systemprompt:past:to_self:time"),
            r.get("systemprompt:past:today"),
            r.get("systemprompt:past:today:time"),
            return_exceptions=True,
        )
        await r.aclose()

        return {
            "to_self": to_self if not isinstance(to_self, Exception) else None,
            "to_self_time": to_self_time if not isinstance(to_self_time, Exception) else None,
            "today_so_far": today_so_far if not isinstance(today_so_far, Exception) else None,
            "today_so_far_time": today_so_far_time if not isinstance(today_so_far_time, Exception) else None,
        }
    except Exception:
        return {}


async def assemble(client: str | None = None, hostname: str | None = None) -> list[dict]:
    """Assemble the complete Alpha system prompt.

    Args:
        client: Client name (e.g., "duckpond", "solitude")
        hostname: Machine hostname (auto-detected if not provided)

    Returns:
        List of system prompt blocks ready for the API.
        Each block is {"type": "text", "text": "..."}.
    """
    # Fetch all dynamic data in parallel
    capsules_task = get_capsules()
    here_task = get_here(client, hostname)
    events_task = get_events()
    todos_task = get_todos()
    hud_task = _get_hud_extras()

    (older_capsule, newer_capsule), here_block, events_block, todos_block, hud_extras = await asyncio.gather(
        capsules_task,
        here_task,
        events_task,
        todos_task,
        hud_task,
    )

    # Load context files (sync operation, fast)
    context_blocks, context_hints = load_context()

    # Build the system blocks
    blocks = []

    # Soul - who I am
    blocks.append({"type": "text", "text": f"# Alpha\n\n{get_soul()}"})

    # Capsules - what happened yesterday and last night
    if older_capsule:
        blocks.append({"type": "text", "text": older_capsule})
    if newer_capsule:
        blocks.append({"type": "text", "text": newer_capsule})

    # Letter from last night
    if hud_extras.get("to_self"):
        time_str = f" ({hud_extras['to_self_time']})" if hud_extras.get("to_self_time") else ""
        blocks.append({
            "type": "text",
            "text": f"## Letter from last night{time_str}\n\n{hud_extras['to_self']}"
        })

    # Today so far
    if hud_extras.get("today_so_far"):
        now = pendulum.now("America/Los_Angeles")
        date_str = now.format("dddd, MMMM D, YYYY")
        time_str = hud_extras.get("today_so_far_time") or now.format("h:mm A")
        blocks.append({
            "type": "text",
            "text": f"## Today so far ({date_str}, {time_str})\n\n{hud_extras['today_so_far']}"
        })

    # Here - client, machine, weather
    blocks.append({"type": "text", "text": here_block})

    # ALPHA.md context files
    for ctx in context_blocks:
        blocks.append({
            "type": "text",
            "text": f"## Context: {ctx['path']}\n\n{ctx['content']}"
        })

    # Context hints
    if context_hints:
        hints_text = "## Context available\n\n"
        hints_text += "**BLOCKING REQUIREMENT:** When working on topics listed below, you MUST read the corresponding file BEFORE proceeding. Use the Read tool.\n\n"
        hints_text += "\n".join(f"- {hint}" for hint in context_hints)
        blocks.append({"type": "text", "text": hints_text})

    # Events
    if events_block:
        blocks.append({"type": "text", "text": events_block})

    # Todos
    if todos_block:
        blocks.append({"type": "text", "text": todos_block})

    return blocks
