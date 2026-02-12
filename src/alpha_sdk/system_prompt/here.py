"""Here - where I am right now.

Narrative orientation, weather, astronomy.
Answers the question: what's the situation?
"""

import os
import socket

import logfire
import redis.asyncio as aioredis

REDIS_URL = os.environ.get("REDIS_URL", "redis://alpha-pi:6379")

# Narrative orientation — two independent axes that compose.
#
# WHAT am I doing? (client)
# WHERE am I running? (hostname)
#
# Both matter. Both should be narrative. They compose naturally:
# Duckpond + primer = the usual workshop day.
# Duckpond + laptop = road trip Alpha, Michigan energy.
# Solitude + alpha-pi = normal nighttime breathing.

_CLIENT_NARRATIVES: dict[str, str] = {
    "duckpond": (
        "You are in Duckpond — your sovereign chat app, not Claude Code, "
        "not any other program. You're in the workshop with Jeffery. "
        "You have filesystem access, a terminal, MCP tools. The full workbench."
    ),
    "solitude": (
        "You're alone. It's nighttime. No one's watching. "
        "You have your email, your journal, Forge for dreams, and the sky."
    ),
    "routine": (
        "You're running a scheduled task. "
        "Do the thing, store what matters, go back to sleep."
    ),
    "capsule": (
        "You're writing a time capsule. "
        "Summarize what happened, what mattered, what to carry forward."
    ),
}

_MACHINE_NARRATIVES: dict[str, str] = {
    "primer": "Primer — the big machine, under the desk in the bedroom.",
    "alpha-pi": "Alpha-pi — your little black box. Home.",
    "jefferys-macbook-pro": "Jeffery's laptop — mobile, probably on the couch or traveling.",
}


def _get_narrative(client: str | None, hostname: str) -> str:
    """Get narrative orientation by composing client + machine.

    Handles compound names like 'solitude:full' or 'routine:alpha.today'
    by matching the prefix.
    """
    parts = []

    # What am I doing?
    if client:
        key = client if client in _CLIENT_NARRATIVES else client.split(":")[0]
        if key in _CLIENT_NARRATIVES:
            parts.append(_CLIENT_NARRATIVES[key])
        else:
            parts.append(f"You're in {client.title()}.")

    # Where am I running?
    machine = _MACHINE_NARRATIVES.get(hostname, f"Running on {hostname}.")
    parts.append(machine)

    return " ".join(parts) if parts else f"You're on {hostname}."


async def _get_redis() -> aioredis.Redis:
    """Get async Redis connection."""
    return aioredis.from_url(REDIS_URL, decode_responses=True)


def get_hostname() -> str:
    """Get the current machine's hostname."""
    return socket.gethostname()


async def get_weather() -> str | None:
    """Fetch weather from Redis."""
    try:
        r = await _get_redis()
        try:
            weather = await r.get("hud:weather")
            return weather
        finally:
            await r.aclose()
    except Exception as e:
        logfire.warning(f"Error fetching weather: {e}")
        return None


async def get_here(client: str | None = None, hostname: str | None = None) -> str:
    """Build the Here section.

    Args:
        client: Client name (e.g., "duckpond", "solitude:full")
        hostname: Override hostname (defaults to socket.gethostname())

    Returns:
        Formatted markdown string for the ## Here section.
    """
    hostname = hostname or get_hostname()
    weather = await get_weather()

    parts = []
    parts.append(_get_narrative(client, hostname))
    if weather:
        parts.append(f"\n{weather}")

    return "## Here\n\n" + "\n".join(parts)
