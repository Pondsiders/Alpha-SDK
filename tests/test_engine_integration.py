"""test_engine_integration.py — Integration tests for the claude subprocess.

These tests spawn a real claude process. They require:
- The `claude` binary on PATH
- A valid authentication (Max subscription or API key)

Run with: uv run pytest tests/test_engine_integration.py -v
Skip with: uv run pytest -m "not integration"

The engine is spawned in a minimal, secure configuration:
- No tools (--tools "")
- No skills (--disable-slash-commands)
- No browser (--no-chrome)
- No MCP servers (--strict-mcp-config with no config)
- Haiku model (cheapest/fastest)
- Trivial system prompt (minimal context)
"""

import pytest

from alpha_sdk.engine import (
    Engine,
    EngineState,
    AssistantEvent,
    InitEvent,
    ResultEvent,
    SystemEvent,
)

# Secure, minimal flags for testing — no tools, no skills, no MCP, no browser.
# This creates the tightest possible sandbox: claude can only converse.
SECURE_TEST_ARGS = [
    "--tools", "",
    "--disable-slash-commands",
    "--no-chrome",
    "--strict-mcp-config",
]


@pytest.fixture
def test_engine():
    """Create a minimal engine configured for integration testing."""
    return Engine(
        model="haiku",
        system_prompt="Reply with exactly one short sentence. No tools.",
        extra_args=SECURE_TEST_ARGS,
    )


@pytest.mark.integration
class TestEngineLifecycle:
    """Test the full engine lifecycle with a real claude process."""

    async def test_start_and_stop(self, test_engine):
        """Engine can start (init handshake) and stop cleanly."""
        try:
            await test_engine.start()

            assert test_engine.state == EngineState.READY
            assert test_engine.pid is not None
            assert test_engine.session_id is None  # Not known until first turn
        finally:
            await test_engine.stop()
            assert test_engine.state == EngineState.STOPPED

    async def test_send_and_receive(self, test_engine):
        """Send a message, receive text response and result."""
        try:
            await test_engine.start()
            await test_engine.send("Say hello.")

            events = []
            async for event in test_engine.events():
                events.append(event)

            # Should have at least one AssistantEvent and one ResultEvent
            assistant_events = [e for e in events if isinstance(e, AssistantEvent)]
            result_events = [e for e in events if isinstance(e, ResultEvent)]

            assert len(assistant_events) >= 1, f"No assistant events. Got: {[type(e).__name__ for e in events]}"
            assert len(result_events) == 1, f"Expected 1 result. Got: {[type(e).__name__ for e in events]}"

            # The assistant should have said something
            text = "".join(e.text for e in assistant_events)
            assert len(text) > 0, "Assistant response was empty"

            # Result should have a session ID
            result = result_events[0]
            assert result.session_id, "No session ID in result"
            assert result.is_error is False

        finally:
            await test_engine.stop()

    async def test_session_id_discovered(self, test_engine):
        """Engine discovers session_id from init or first turn."""
        try:
            await test_engine.start()
            # Session ID might not be known until the first turn
            await test_engine.send("Say hi.")
            async for event in test_engine.events():
                if isinstance(event, ResultEvent):
                    break
            # After first turn, session_id should be discovered
            assert test_engine.session_id is not None
            assert len(test_engine.session_id) > 0
        finally:
            await test_engine.stop()
