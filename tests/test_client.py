"""Tests for client.py — AlphaClient composer."""

import pytest

from alpha_sdk.client import AlphaClient
from alpha_sdk.engine import Engine
from alpha_sdk.router import Router
from alpha_sdk.session import Session


class TestAlphaClientConstruction:
    def test_default_construction(self):
        client = AlphaClient()
        assert client.engine is not None
        assert client.session is not None
        assert isinstance(client.engine, Engine)
        assert isinstance(client.session, Session)

    def test_custom_model(self):
        client = AlphaClient(model="claude-haiku-4-20250514")
        assert client.engine.model == "claude-haiku-4-20250514"

    def test_with_system_prompt(self):
        client = AlphaClient(system_prompt="Be helpful.")
        assert client.engine.system_prompt == "Be helpful."

    def test_with_observers(self):
        class DummyObserver:
            async def on_event(self, event):
                pass

        obs = DummyObserver()
        client = AlphaClient(observers=[obs])
        assert client.session.router.observer_count == 1

    def test_session_id_starts_none(self):
        client = AlphaClient()
        assert client.session_id is None

    def test_queue_maxsize(self):
        client = AlphaClient(queue_maxsize=10)
        # Queue is created internally — verify via session
        assert client.session.queue is not None
