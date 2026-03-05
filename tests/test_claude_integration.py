"""test_claude_integration.py — Live tests for the Claude subprocess.

Spawns a real claude process with Haiku. Costs real tokens (~$0.001/test).

Run:  uv run pytest tests/test_claude_integration.py -v
Skip: uv run pytest -m "not integration"
"""

import pytest

from alpha_sdk.claude import (
    Claude,
    ClaudeState,
    AssistantEvent,
    ResultEvent,
    StreamEvent,
)


@pytest.fixture
def test_claude():
    """Minimal Claude configured for integration testing."""
    return Claude(
        model="claude-haiku-4-5-20251001",
        system_prompt="Reply with exactly one short sentence. No tools.",
    )


@pytest.mark.integration
class TestClaudeLifecycle:
    """Full lifecycle: start → send → events → stop."""

    async def test_start_and_stop(self, test_claude):
        """Init handshake completes, clean shutdown."""
        try:
            await test_claude.start()
            assert test_claude.state == ClaudeState.READY
            assert test_claude.pid is not None
        finally:
            await test_claude.stop()
            assert test_claude.state == ClaudeState.STOPPED

    async def test_send_receive_turn(self, test_claude):
        """Send a message, get assistant text + result."""
        try:
            await test_claude.start()
            await test_claude.send([{"type": "text", "text": "Say hello."}])

            got_assistant = False
            got_result = False
            got_stream = False
            assistant_text = ""

            async for event in test_claude.events():
                if isinstance(event, StreamEvent) and event.delta_text:
                    got_stream = True
                elif isinstance(event, AssistantEvent):
                    got_assistant = True
                    # Haiku 4.5 uses extended thinking — first AssistantEvent
                    # may contain only a thinking block (no text). Accumulate
                    # across all AssistantEvents before checking.
                    assistant_text += event.text
                elif isinstance(event, ResultEvent):
                    got_result = True
                    assert event.session_id, "No session ID in result"
                    assert event.is_error is False
                    break

            assert got_assistant, "No AssistantEvent received"
            assert got_result, "No ResultEvent received"
            assert got_stream, "No streaming deltas received"
            assert len(assistant_text) > 0, "No text in any AssistantEvent (only thinking blocks?)"

        finally:
            await test_claude.stop()

    async def test_session_id_discovered(self, test_claude):
        """Session ID is populated after first turn."""
        try:
            await test_claude.start()
            assert test_claude.session_id is None

            await test_claude.send([{"type": "text", "text": "Hi."}])
            async for event in test_claude.events():
                if isinstance(event, ResultEvent):
                    break

            assert test_claude.session_id is not None
            assert len(test_claude.session_id) > 0
        finally:
            await test_claude.stop()

    async def test_multi_turn(self, test_claude):
        """Two turns in sequence — claude stays alive."""
        try:
            await test_claude.start()

            # Turn 1
            await test_claude.send([{"type": "text", "text": "Say 'one'."}])
            async for event in test_claude.events():
                if isinstance(event, ResultEvent):
                    break

            # Turn 2
            await test_claude.send([{"type": "text", "text": "Say 'two'."}])
            turn2_text = ""
            async for event in test_claude.events():
                if isinstance(event, AssistantEvent):
                    turn2_text = event.text
                elif isinstance(event, ResultEvent):
                    assert event.num_turns >= 1
                    break

            assert len(turn2_text) > 0, "Second turn produced no text"

        finally:
            await test_claude.stop()

    async def test_token_count_from_proxy(self, test_claude):
        """Proxy sniffs token count from API responses."""
        try:
            await test_claude.start()
            assert test_claude.token_count == 0

            await test_claude.send([{"type": "text", "text": "Hi."}])
            async for event in test_claude.events():
                if isinstance(event, ResultEvent):
                    break

            assert test_claude.token_count > 0, (
                "Token count still 0 after a turn. "
                "Proxy may not be sniffing SSE correctly."
            )
        finally:
            await test_claude.stop()
