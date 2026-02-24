"""Hand-off tool — Alpha's bedtime routine button.

Alpha calls this when she's ready to transition her context. The tool
sets a flag; after the current turn's response stream finishes, the
client automatically sends /compact with her instructions, then wakes
her up in a fresh context with orientation and memories.

The tool requires a memory — you can't go to sleep without first
remembering something. The last thing you do before lights-out
should always be storing what mattered.

Auto-compact is the fire alarm. Hand-off is the bedtime routine.
"""

from typing import Any, Callable, Coroutine

from claude_agent_sdk import tool, create_sdk_mcp_server


def create_handoff_server(
    on_handoff: Callable[[str], None],
    store_memory: Callable[[str], Coroutine[Any, Any, dict]],
):
    """Create the hand-off MCP server.

    Args:
        on_handoff: Callback that receives compact instructions.
                    Should call AlphaClient.request_compact().
        store_memory: Async function to store a memory in Cortex.
                      Should call cortex.store().

    Returns:
        MCP server configuration dict
    """

    @tool(
        "handoff",
        "Hand off your context. Call this when you're ready to gracefully "
        "transition to a fresh context window. Pass instructions telling "
        "the summarizer what to focus on — what's still in progress, "
        "what's finished, what matters most for future-you.",
        {
            "type": "object",
            "properties": {
                "instructions": {
                    "type": "string",
                    "description": (
                        "Instructions for the compaction summarizer. "
                        "What to focus on, what's done, what's in progress. "
                        "Example: 'The fetch tool is done. Focus on the "
                        "hand-off mechanism — we were testing the flag.'"
                    ),
                },
                "memory": {
                    "type": "string",
                    "description": (
                        "Your last memory of this context window. "
                        "What matters right now? What would you want "
                        "future-you to find? You can't hand off without "
                        "first remembering."
                    ),
                },
            },
            "required": ["instructions", "memory"],
        },
    )
    async def handoff(args: dict[str, Any]) -> dict[str, Any]:
        """Store a last memory, then flag the compact."""
        instructions = args["instructions"]
        memory = args["memory"]

        # Store the last memory before lights-out
        result = await store_memory(memory)
        memory_id = result.get("id", "?")

        # Flag the compact
        on_handoff(instructions)

        return {
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Memory #{memory_id} stored. "
                        "Hand-off flagged. When this turn ends, "
                        "/compact will fire with your instructions, "
                        "then you'll wake up in a fresh context. "
                        "Last thoughts — say what you need to say."
                    ),
                }
            ]
        }

    return create_sdk_mcp_server(
        name="handoff",
        version="1.0.0",
        tools=[handoff],
    )
