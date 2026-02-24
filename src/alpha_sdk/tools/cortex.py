"""Cortex memory tools - native MCP server for alpha_sdk.

Direct Postgres access—no HTTP layer, no Cortex service dependency.
The store() tool clears the pending memorables buffer as a side effect,
closing the feedback loop with Intro.

Usage:
    from alpha_sdk.tools import create_cortex_server

    # In AlphaClient, get session ID dynamically
    mcp_servers = {
        "cortex": create_cortex_server(get_session_id=lambda: client.session_id)
    }
"""

from typing import Any, Callable

from claude_agent_sdk import tool, create_sdk_mcp_server

from ..memories import store as cortex_store, search as cortex_search, recent as cortex_recent


def create_cortex_server(
    get_session_id: Callable[[], str | None] | None = None,
    clear_memorables: Callable[[], int] | None = None,
):
    """Create the Cortex MCP server.

    Args:
        get_session_id: Optional callable that returns the current session ID.
        clear_memorables: Optional callable that clears pending memorables and
                         returns the count cleared. Provided by AlphaClient.

    Returns:
        MCP server configuration dict
    """

    @tool(
        "store",
        "Store a memory in Cortex. Use this to remember important moments, realizations, or anything worth preserving.",
        {
            "type": "object",
            "properties": {
                "memory": {"type": "string", "description": "The memory text content"},
                "image": {"type": "string", "description": "Optional path to an image to attach (will be thumbnailed to 768px JPEG)"},
            },
            "required": ["memory"],
        },
    )
    async def store_memory(args: dict[str, Any]) -> dict[str, Any]:
        """Store a memory and clear the memorables buffer.

        Args (via tool call):
            memory: The memory text content
            image: Optional path to an image to attach (will be thumbnailed to 768px JPEG)
        """
        memory = args["memory"]
        image = args.get("image")

        try:
            result = await cortex_store(memory, image=image)

            if result is None:
                return {"content": [{"type": "text", "text": "Error storing memory"}]}

            memory_id = result.get("id", "unknown")

            # Clear the memorables buffer - this is the feedback mechanism
            cleared = clear_memorables() if clear_memorables else 0

            # Build response
            response_text = f"Memory stored (id: {memory_id})"
            if result.get("thumbnail_path"):
                response_text += f" [image: {result['thumbnail_path']}]"
            if cleared > 0:
                response_text += f" - cleared {cleared} pending suggestion(s)"

            return {"content": [{"type": "text", "text": response_text}]}

        except Exception as e:
            return {"content": [{"type": "text", "text": f"Error storing memory: {e}"}]}

    @tool(
        "search",
        "Search memories in Cortex. Returns semantically similar memories. Limit defaults to 5.",
        {"query": str}
    )
    async def search_memories(args: dict[str, Any]) -> dict[str, Any]:
        """Search for memories matching a query."""
        query = args["query"]
        limit = args.get("limit", 5)

        try:
            memories = await cortex_search(query, limit=limit)

            if not memories:
                return {"content": [{"type": "text", "text": "No memories found."}]}

            # Format results
            lines = [f"Found {len(memories)} memor{'y' if len(memories) == 1 else 'ies'}:\n"]
            for mem in memories:
                score = mem.get("score", 0)
                content = mem.get("content", "")
                created = mem.get("created_at", "")[:10]  # Just the date
                image_flag = " 📷" if mem.get("image_path") else ""
                lines.append(f"[{score:.2f}] ({created}{image_flag}) {content}\n")

            return {"content": [{"type": "text", "text": "\n".join(lines)}]}

        except Exception as e:
            return {"content": [{"type": "text", "text": f"Error searching memories: {e}"}]}

    @tool(
        "recent",
        "Get recent memories from Cortex. Limit defaults to 10.",
        {}
    )
    async def recent_memories(args: dict[str, Any]) -> dict[str, Any]:
        """Get the most recent memories."""
        limit = args.get("limit", 10)

        try:
            memories = await cortex_recent(limit=limit)

            if not memories:
                return {"content": [{"type": "text", "text": "No recent memories."}]}

            # Format results
            lines = [f"Last {len(memories)} memor{'y' if len(memories) == 1 else 'ies'}:\n"]
            for mem in memories:
                content = mem.get("content", "")
                created = mem.get("created_at", "")[:16]  # Date and time
                image_flag = " 📷" if mem.get("image_path") else ""
                lines.append(f"({created}{image_flag}) {content}\n")

            return {"content": [{"type": "text", "text": "\n".join(lines)}]}

        except Exception as e:
            return {"content": [{"type": "text", "text": f"Error getting recent memories: {e}"}]}

    # Bundle into MCP server
    return create_sdk_mcp_server(
        name="cortex",
        version="2.0.0",  # Bumped version - now using direct Postgres
        tools=[store_memory, search_memories, recent_memories]
    )
