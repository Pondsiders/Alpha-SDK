"""test_mcp_dispatch.py — Tests for in-process MCP dispatch.

Tests the MCP config merging, JSON-RPC dispatch to FastMCP servers,
and the three-way control_request handler. No subprocess — pure logic.
"""

import json

import pytest
from mcp.server.fastmcp import FastMCP

from alpha_sdk.claude import Claude, _ControlRequestEvent


# -- Test fixtures -----------------------------------------------------------


def _make_frobozz() -> FastMCP:
    """A minimal FastMCP server with one tool."""
    server = FastMCP("frobozz")

    @server.tool()
    def echo(message: str) -> str:
        """Echo a message back."""
        return f"Frobozz says: {message}"

    return server


def _make_control_event(
    subtype: str,
    request_id: str = "req_test_001",
    **extra,
) -> _ControlRequestEvent:
    """Build a _ControlRequestEvent for testing."""
    request = {"subtype": subtype, **extra}
    return _ControlRequestEvent(
        raw={"type": "control_request", "request_id": request_id, "request": request},
        request_id=request_id,
        tool_name=extra.get("tool_name", subtype),
        request=request,
    )


# -- MCP config merging -----------------------------------------------------


class TestBuildMcpConfig:
    """_build_mcp_config merges consumer + SDK servers into inline JSON."""

    def test_sdk_servers_only(self):
        """SDK servers produce type=sdk entries."""
        frobozz = _make_frobozz()
        claude = Claude(mcp_servers={"frobozz": frobozz})

        config = claude._build_mcp_config()
        assert config is not None

        parsed = json.loads(config)
        servers = parsed["mcpServers"]
        assert "frobozz" in servers
        assert servers["frobozz"]["type"] == "sdk"
        assert servers["frobozz"]["name"] == "frobozz"

    def test_consumer_inline_json_only(self):
        """Consumer inline JSON is passed through without SDK additions."""
        consumer_config = json.dumps({
            "mcpServers": {
                "external": {"command": "some-server", "args": ["--port", "3000"]},
            }
        })
        claude = Claude(mcp_config=consumer_config)

        config = claude._build_mcp_config()
        parsed = json.loads(config)
        assert "external" in parsed["mcpServers"]

    def test_merge_consumer_and_sdk(self):
        """Both consumer and SDK servers appear in merged config."""
        consumer_config = json.dumps({
            "mcpServers": {
                "external": {"command": "ext-server"},
            }
        })
        frobozz = _make_frobozz()
        claude = Claude(
            mcp_config=consumer_config,
            mcp_servers={"frobozz": frobozz},
        )

        config = claude._build_mcp_config()
        parsed = json.loads(config)
        assert "external" in parsed["mcpServers"]
        assert "frobozz" in parsed["mcpServers"]
        assert parsed["mcpServers"]["frobozz"]["type"] == "sdk"

    def test_no_servers_returns_none(self):
        """No consumer config + no SDK servers → None (no --mcp-config flag)."""
        claude = Claude()
        assert claude._build_mcp_config() is None

    def test_sdk_server_overwrites_consumer_collision(self):
        """SDK server names take precedence over consumer names."""
        consumer_config = json.dumps({
            "mcpServers": {
                "cortex": {"command": "fake-cortex"},
            }
        })
        frobozz = _make_frobozz()
        claude = Claude(
            mcp_config=consumer_config,
            mcp_servers={"cortex": frobozz},
        )

        config = claude._build_mcp_config()
        parsed = json.loads(config)
        # SDK server wins — type "sdk", not the consumer's command entry
        assert parsed["mcpServers"]["cortex"]["type"] == "sdk"

    def test_consumer_file_path_nonexistent(self):
        """Non-existent file path is silently ignored."""
        claude = Claude(
            mcp_config="/nonexistent/path/to/config.json",
            mcp_servers={"frobozz": _make_frobozz()},
        )
        config = claude._build_mcp_config()
        parsed = json.loads(config)
        # Only SDK server present — file was ignored
        assert len(parsed["mcpServers"]) == 1
        assert "frobozz" in parsed["mcpServers"]


# -- MCP dispatch (JSON-RPC routing) ----------------------------------------


class TestDispatchMcp:
    """_dispatch_mcp routes JSON-RPC to FastMCP request_handlers."""

    @pytest.fixture
    def claude_with_frobozz(self):
        return Claude(mcp_servers={"frobozz": _make_frobozz()})

    async def test_initialize(self, claude_with_frobozz):
        """initialize returns hardcoded protocol response."""
        result = await claude_with_frobozz._dispatch_mcp("frobozz", {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {},
        })
        assert result["id"] == 1
        assert result["result"]["protocolVersion"] == "2024-11-05"
        assert result["result"]["serverInfo"]["name"] == "frobozz"

    async def test_notifications_initialized(self, claude_with_frobozz):
        """notifications/initialized is a no-op ack."""
        result = await claude_with_frobozz._dispatch_mcp("frobozz", {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        })
        assert result["result"] == {}

    async def test_tools_list(self, claude_with_frobozz):
        """tools/list returns registered tool definitions."""
        result = await claude_with_frobozz._dispatch_mcp("frobozz", {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        })
        tools = result["result"]["tools"]
        names = [t["name"] for t in tools]
        assert "echo" in names

    async def test_tools_call(self, claude_with_frobozz):
        """tools/call invokes the tool and returns content."""
        result = await claude_with_frobozz._dispatch_mcp("frobozz", {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "echo", "arguments": {"message": "quack"}},
        })
        content = result["result"]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert "Frobozz says: quack" in content[0]["text"]

    async def test_unknown_server(self, claude_with_frobozz):
        """Unknown server name returns JSON-RPC error."""
        result = await claude_with_frobozz._dispatch_mcp("nonexistent", {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/list",
        })
        assert "error" in result
        assert result["error"]["code"] == -32601
        assert "nonexistent" in result["error"]["message"]

    async def test_unknown_method(self, claude_with_frobozz):
        """Unknown method returns JSON-RPC method-not-found error."""
        result = await claude_with_frobozz._dispatch_mcp("frobozz", {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "resources/list",
        })
        assert "error" in result
        assert result["error"]["code"] == -32601

    async def test_tool_call_bad_name(self, claude_with_frobozz):
        """Calling a nonexistent tool returns error in content."""
        result = await claude_with_frobozz._dispatch_mcp("frobozz", {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {"name": "nonexistent_tool", "arguments": {}},
        })
        # FastMCP returns unknown tool errors as text content
        content = result["result"]["content"]
        assert any("Unknown tool" in c.get("text", "") for c in content)


# -- Three-way control_request handler --------------------------------------


class TestHandleControlRequest:
    """_handle_control_request routes by subtype."""

    async def test_permission_with_handler_approved(self):
        """Permission request + handler → calls handler, sends response."""
        responses = []

        async def approve_all(req: dict) -> bool:
            return True

        claude = Claude(permission_handler=approve_all)
        # Mock _send_json to capture what would be sent
        claude._send_json = lambda obj: responses.append(obj) or _async_noop()

        event = _make_control_event("tool_permission", tool_name="Bash")
        await claude._handle_control_request(event)

        assert len(responses) == 1
        resp = responses[0]["response"]
        assert resp["response"]["approved"] is True
        assert resp["request_id"] == "req_test_001"

    async def test_permission_with_handler_denied(self):
        """Permission handler can deny — sends approved=False."""
        responses = []

        async def deny_all(req: dict) -> bool:
            return False

        claude = Claude(permission_handler=deny_all)
        claude._send_json = lambda obj: responses.append(obj) or _async_noop()

        event = _make_control_event("tool_permission", tool_name="Bash")
        await claude._handle_control_request(event)

        assert responses[0]["response"]["response"]["approved"] is False

    async def test_permission_without_handler_raises(self):
        """No handler + permission request → RuntimeError (fail loud)."""
        claude = Claude()  # No permission_handler

        event = _make_control_event("tool_permission", tool_name="Bash")
        with pytest.raises(RuntimeError, match="no permission_handler"):
            await claude._handle_control_request(event)

    async def test_mcp_message_dispatched(self):
        """MCP message → dispatched to in-process server."""
        responses = []
        frobozz = _make_frobozz()
        claude = Claude(mcp_servers={"frobozz": frobozz})
        claude._send_json = lambda obj: responses.append(obj) or _async_noop()

        event = _make_control_event(
            "mcp_message",
            server_name="frobozz",
            message={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {},
            },
        )
        await claude._handle_control_request(event)

        assert len(responses) == 1
        resp = responses[0]["response"]
        assert resp["subtype"] == "success"
        mcp_resp = resp["response"]["mcp_response"]
        assert mcp_resp["result"]["protocolVersion"] == "2024-11-05"

    async def test_mcp_tool_call_dispatched(self):
        """MCP tools/call → tool executes, result wrapped correctly."""
        responses = []
        frobozz = _make_frobozz()
        claude = Claude(mcp_servers={"frobozz": frobozz})
        claude._send_json = lambda obj: responses.append(obj) or _async_noop()

        event = _make_control_event(
            "mcp_message",
            server_name="frobozz",
            message={
                "jsonrpc": "2.0",
                "id": 42,
                "method": "tools/call",
                "params": {"name": "echo", "arguments": {"message": "hello"}},
            },
        )
        await claude._handle_control_request(event)

        mcp_resp = responses[0]["response"]["response"]["mcp_response"]
        assert mcp_resp["id"] == 42
        content = mcp_resp["result"]["content"]
        assert any("Frobozz says: hello" in c.get("text", "") for c in content)


# -- Helpers -----------------------------------------------------------------


async def _async_noop():
    """Awaitable no-op for mocking _send_json."""
    pass
