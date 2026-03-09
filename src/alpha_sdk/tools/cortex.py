"""Cortex memory tools — FastMCP server for alpha_sdk.

Direct Postgres access — no HTTP layer, no Cortex service dependency.
The store() tool optionally clears the pending memorables buffer,
closing the feedback loop with Intro.

Usage:
    from alpha_sdk.tools.cortex import create_cortex_server

    server = create_cortex_server()
    # Pass to Claude(mcp_servers={"cortex": server})
"""

from typing import Callable

from mcp.server.fastmcp import FastMCP

from ..memories.cortex import (
    store as cortex_store,
    search as cortex_search,
    recent as cortex_recent,
    get as cortex_get,
)


def create_cortex_server(
    clear_memorables: Callable[[], int] | None = None,
) -> FastMCP:
    """Create the Cortex MCP server.

    Args:
        clear_memorables: Optional callable that clears pending memorables and
                         returns the count cleared. Provided by the consumer
                         when the suggest pipeline is wired up.

    Returns:
        FastMCP server instance ready for dispatch
    """

    server = FastMCP("cortex")

    @server.tool(
        description=(
            "Store a memory in Cortex. Use this to remember important moments, "
            "realizations, or anything worth preserving."
        ),
    )
    async def store(memory: str, image: str | None = None) -> str:
        """Store a memory and optionally clear the memorables buffer."""
        result = await cortex_store(memory, image=image)

        if result is None:
            return "Error storing memory"

        memory_id = result.get("id", "unknown")

        # Clear the memorables buffer — feedback mechanism with Intro
        cleared = clear_memorables() if clear_memorables else 0

        # Build response
        response = f"Memory stored (id: {memory_id})"
        if result.get("thumbnail_path"):
            response += f" [image: {result['thumbnail_path']}]"
        if cleared > 0:
            response += f" - cleared {cleared} pending suggestion(s)"

        return response

    @server.tool(
        description=(
            "Search memories in Cortex. Returns semantically similar memories. "
            "Limit defaults to 5."
        ),
    )
    async def search(query: str) -> str:
        """Search for memories matching a query."""
        memories = await cortex_search(query, limit=5)

        if not memories:
            return "No memories found."

        # Format results
        lines = [f"Found {len(memories)} memor{'y' if len(memories) == 1 else 'ies'}:\n"]
        for mem in memories:
            score = mem.get("score", 0)
            content = mem.get("content", "")
            created = mem.get("created_at", "")[:10]  # Just the date
            image_flag = " [img]" if mem.get("image_path") else ""
            lines.append(f"[{score:.2f}] ({created}{image_flag}) {content}\n")

        return "\n".join(lines)

    @server.tool(
        description="Get recent memories from Cortex. Limit defaults to 10.",
    )
    async def recent() -> str:
        """Get the most recent memories."""
        memories = await cortex_recent(limit=10)

        if not memories:
            return "No recent memories."

        # Format results
        lines = [f"Last {len(memories)} memor{'y' if len(memories) == 1 else 'ies'}:\n"]
        for mem in memories:
            content = mem.get("content", "")
            created = mem.get("created_at", "")[:16]  # Date and time
            image_flag = " [img]" if mem.get("image_path") else ""
            lines.append(f"({created}{image_flag}) {content}\n")

        return "\n".join(lines)

    @server.tool(
        description="Get a specific memory by its ID.",
    )
    async def get(memory_id: int) -> str:
        """Retrieve a single memory by ID."""
        mem = await cortex_get(memory_id)

        if mem is None:
            return f"Memory {memory_id} not found."

        content = mem.get("content", "")
        created = mem.get("created_at", "")
        tags = mem.get("tags")
        image_flag = f"\n[image: {mem['image_path']}]" if mem.get("image_path") else ""

        result = f"Memory {memory_id} ({created}):\n{content}"
        if tags:
            result += f"\nTags: {', '.join(tags)}"
        result += image_flag

        return result

    return server
