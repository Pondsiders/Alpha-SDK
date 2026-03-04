"""test_proxy_forwarding.py — Tests for proxy upstream URL forwarding.

Verifies that _Proxy forwards requests to the configured upstream_url
instead of hardcoded api.anthropic.com. Catches regressions if someone
reverts the upstream_url parameter.

The existing test_proxy.py covers pure rewrite logic (no HTTP).
This file covers the HTTP forwarding behavior of the proxy server.
"""

import json
import socket

import httpx
import pytest
from aiohttp import web

from alpha_sdk.proxy import ANTHROPIC_API_URL, _Proxy


# -- Helpers ------------------------------------------------------------------


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# Minimal valid Anthropic SSE response — enough to exercise the proxy's
# streaming and token sniffing without hitting the real API.
FAKE_SSE_RESPONSE = (
    "event: message_start\n"
    'data: {"type":"message_start","message":{"usage":'
    '{"input_tokens":100,"cache_creation_input_tokens":0,'
    '"cache_read_input_tokens":0}}}\n\n'
    "event: content_block_start\n"
    'data: {"type":"content_block_start","index":0,'
    '"content_block":{"type":"text","text":""}}\n\n'
    "event: content_block_delta\n"
    'data: {"type":"content_block_delta","index":0,'
    '"delta":{"type":"text_delta","text":"Hello from fake upstream!"}}\n\n'
    "event: content_block_stop\n"
    'data: {"type":"content_block_stop","index":0}\n\n'
    "event: message_delta\n"
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
    '"usage":{"output_tokens":5}}\n\n'
    "event: message_stop\n"
    'data: {"type":"message_stop"}\n\n'
)


# -- Fixtures -----------------------------------------------------------------


@pytest.fixture
async def fake_upstream():
    """A minimal HTTP server that records requests and returns valid SSE.

    Yields a dict with:
        url:      base URL of the fake server
        received: list of recorded requests (mutated in place)
    """
    received = []

    async def handle(request: web.Request):
        body_bytes = await request.read()
        received.append({
            "method": request.method,
            "path": request.path,
            "body": json.loads(body_bytes) if body_bytes else None,
            "headers": {k: v for k, v in request.headers.items()},
        })
        return web.Response(
            body=FAKE_SSE_RESPONSE,
            content_type="text/event-stream",
        )

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", handle)

    port = _find_free_port()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()

    yield {"url": f"http://127.0.0.1:{port}", "received": received}

    await runner.cleanup()


# -- Unit tests: upstream_url configuration -----------------------------------


class TestProxyUpstreamConfig:
    """Verify _Proxy stores the correct upstream URL."""

    def test_default_upstream_is_anthropic(self):
        """Without upstream_url, proxy targets api.anthropic.com."""
        proxy = _Proxy()
        assert proxy._upstream_url == ANTHROPIC_API_URL

    def test_none_upstream_is_anthropic(self):
        """Explicit None falls back to api.anthropic.com."""
        proxy = _Proxy(upstream_url=None)
        assert proxy._upstream_url == ANTHROPIC_API_URL

    def test_custom_upstream(self):
        """Custom upstream_url is stored."""
        proxy = _Proxy(upstream_url="http://localhost:9999")
        assert proxy._upstream_url == "http://localhost:9999"

    def test_empty_string_upstream_is_anthropic(self):
        """Empty string is falsy — falls back to api.anthropic.com."""
        proxy = _Proxy(upstream_url="")
        assert proxy._upstream_url == ANTHROPIC_API_URL


# -- Integration tests: actual HTTP forwarding --------------------------------


class TestProxyForwarding:
    """Integration: verify requests actually reach the configured upstream."""

    async def test_forwards_to_custom_upstream(self, fake_upstream):
        """POST through the proxy reaches the custom upstream, not Anthropic."""
        proxy = _Proxy(upstream_url=fake_upstream["url"])
        proxy_port = await proxy.start()

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"http://127.0.0.1:{proxy_port}/v1/messages",
                    json={
                        "model": "claude-test",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                    headers={
                        "content-type": "application/json",
                        "x-api-key": "test-key-123",
                    },
                )

            # The fake upstream received exactly one request
            assert len(fake_upstream["received"]) == 1
            req = fake_upstream["received"][0]
            assert req["method"] == "POST"
            assert req["path"] == "/v1/messages"
            assert req["body"]["model"] == "claude-test"

            # Proxy forwarded the auth header
            assert req["headers"].get("x-api-key") == "test-key-123"

            # Proxy returned the upstream's SSE response
            assert response.status_code == 200

        finally:
            await proxy.stop()

    async def test_health_endpoint_does_not_hit_upstream(self, fake_upstream):
        """The /health endpoint is handled locally — no upstream call."""
        proxy = _Proxy(upstream_url=fake_upstream["url"])
        await proxy.start()

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://127.0.0.1:{proxy.port}/health"
                )

            assert response.status_code == 200
            assert response.text == "ok"
            assert len(fake_upstream["received"]) == 0

        finally:
            await proxy.stop()

    async def test_sniffs_token_count_from_sse(self, fake_upstream):
        """Proxy extracts input_tokens from the upstream's SSE stream."""
        proxy = _Proxy(upstream_url=fake_upstream["url"])
        await proxy.start()

        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"http://127.0.0.1:{proxy.port}/v1/messages",
                    json={"messages": [{"role": "user", "content": "Hello"}]},
                    headers={"content-type": "application/json"},
                )

            # Token count sniffed from message_start: input_tokens=100
            assert proxy.token_count == 100

        finally:
            await proxy.stop()

    async def test_preserves_path(self, fake_upstream):
        """Proxy preserves the full request path when forwarding."""
        proxy = _Proxy(upstream_url=fake_upstream["url"])
        await proxy.start()

        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"http://127.0.0.1:{proxy.port}/v1/messages",
                    json={"messages": []},
                    headers={"content-type": "application/json"},
                )

            assert fake_upstream["received"][0]["path"] == "/v1/messages"

        finally:
            await proxy.stop()
