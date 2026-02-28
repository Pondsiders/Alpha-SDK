"""Tests for producers/human.py — human input producer."""

import asyncio

import pytest

from alpha_sdk.producers.human import HumanInput


# -- Mock session for testing ------------------------------------------------


class MockSession:
    """Minimal session mock for testing producers."""

    def __init__(self):
        self.messages: list[tuple[str, str]] = []  # (content, producer)

    async def send(self, content: str, producer: str = "") -> None:
        self.messages.append((content, producer))


# -- Tests -------------------------------------------------------------------


class TestHumanInput:
    @pytest.mark.asyncio
    async def test_sends_input_to_session(self):
        inputs = iter(["hello", "world", None])

        async def fake_input():
            return next(inputs)

        session = MockSession()
        human = HumanInput(input_fn=fake_input)
        await human.run(session)

        assert len(session.messages) == 2
        assert session.messages[0] == ("hello", "human")
        assert session.messages[1] == ("world", "human")

    @pytest.mark.asyncio
    async def test_skips_empty_lines(self):
        inputs = iter(["hello", "", "  ", "world", None])

        async def fake_input():
            return next(inputs)

        session = MockSession()
        human = HumanInput(input_fn=fake_input)
        await human.run(session)

        assert len(session.messages) == 2

    @pytest.mark.asyncio
    async def test_stops_on_eof(self):
        async def immediate_eof():
            return None

        session = MockSession()
        human = HumanInput(input_fn=immediate_eof)
        await human.run(session)

        assert len(session.messages) == 0

    @pytest.mark.asyncio
    async def test_custom_producer_name(self):
        inputs = iter(["test", None])

        async def fake_input():
            return next(inputs)

        session = MockSession()
        human = HumanInput(input_fn=fake_input, name="cli")
        await human.run(session)

        assert session.messages[0][1] == "cli"
