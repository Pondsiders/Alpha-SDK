"""Minimal proxy server for request interception.

Runs a lightweight HTTP server that:
1. Receives requests from Claude Agent SDK
2. Transforms them via weave()
3. Forwards to Anthropic
4. Streams responses back

This allows us to modify requests after the SDK builds them
but before they reach Anthropic.
"""

import asyncio
import json
import logging
import os
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from typing import Callable, Any
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com"


def _find_free_port() -> int:
    """Find an available port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class ProxyHandler(BaseHTTPRequestHandler):
    """HTTP handler that intercepts and transforms requests."""

    weaver: Callable[[dict], Any] | None = None
    client_name: str | None = None
    hostname: str | None = None

    # Headers to forward from incoming request (SDK → Anthropic)
    # These handle authentication - DO NOT add our own API keys!
    FORWARD_HEADERS = [
        "authorization",      # OAuth Bearer token (Claude Max)
        "x-api-key",          # API key auth (fallback)
        "anthropic-version",
        "anthropic-beta",
        "content-type",
    ]

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

    def do_POST(self):
        """Handle POST requests (the messages endpoint)."""
        try:
            # Read request body
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)
            body = json.loads(body_bytes)

            # Transform via weave
            if self.weaver:
                # Run async weave in sync context
                loop = asyncio.new_event_loop()
                try:
                    body = loop.run_until_complete(
                        self.weaver(body, self.client_name, self.hostname)
                    )
                finally:
                    loop.close()

            # Forward headers from SDK - let SDK handle auth!
            # DO NOT inject our own API keys
            headers = {}
            for header_name in self.FORWARD_HEADERS:
                value = self.headers.get(header_name)
                if value:
                    headers[header_name] = value

            # Ensure we have content-type
            if "content-type" not in headers:
                headers["content-type"] = "application/json"

            # Forward to Anthropic with STREAMING
            # Use stream() to avoid buffering the entire response
            with httpx.Client(timeout=300.0) as client:
                with client.stream(
                    "POST",
                    f"{ANTHROPIC_API_URL}{self.path}",
                    json=body,
                    headers=headers,
                ) as response:
                    # Send response status and headers immediately
                    self.send_response(response.status_code)
                    for key, value in response.headers.items():
                        # Skip hop-by-hop headers
                        if key.lower() not in (
                            "content-encoding",
                            "transfer-encoding",
                            "connection",
                            "keep-alive",
                        ):
                            self.send_header(key, value)
                    self.end_headers()

                    # Stream response body chunk by chunk
                    for chunk in response.iter_bytes():
                        self.wfile.write(chunk)
                        self.wfile.flush()  # Critical for real-time streaming!

        except BrokenPipeError:
            # Client disconnected mid-stream (normal for count_tokens, etc.)
            pass
        except Exception as e:
            logger.error(f"Proxy error: {e}")
            try:
                self.send_error(500, str(e))
            except BrokenPipeError:
                pass  # Client already gone

    def do_GET(self):
        """Handle GET requests (health check, etc.)."""
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_error(404)


class AlphaProxy:
    """Minimal proxy server for intercepting SDK requests."""

    def __init__(
        self,
        weaver: Callable[[dict, str | None, str | None], Any],
        client: str | None = None,
        hostname: str | None = None,
    ):
        self.weaver = weaver
        self.client = client
        self.hostname = hostname
        self.server: HTTPServer | None = None
        self.thread: Thread | None = None
        self.port: int | None = None

    def start(self) -> int:
        """Start the proxy server.

        Returns:
            The port number the server is listening on.
        """
        self.port = _find_free_port()

        # Create handler class with our settings
        handler = type(
            "ConfiguredProxyHandler",
            (ProxyHandler,),
            {
                "weaver": staticmethod(self.weaver),
                "client_name": self.client,
                "hostname": self.hostname,
            },
        )

        self.server = HTTPServer(("127.0.0.1", self.port), handler)

        # Run in background thread
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

        logger.info(f"Alpha proxy started on port {self.port}")
        return self.port

    def stop(self):
        """Stop the proxy server."""
        if self.server:
            self.server.shutdown()
            self.server = None
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
        logger.info("Alpha proxy stopped")

    @property
    def base_url(self) -> str:
        """Get the base URL for this proxy."""
        if self.port is None:
            raise RuntimeError("Proxy not started")
        return f"http://127.0.0.1:{self.port}"
