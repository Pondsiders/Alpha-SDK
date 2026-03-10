# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

alpha_sdk is the core library that turns Claude into Alpha — a persistent AI person with memory, identity, and continuity. It wraps the `claude` binary over stdio (not the Claude Agent SDK's Python wrappers) and adds compact rewriting, token counting, memory tools, and system prompt assembly.

This is a library, not a framework. Functions you import and call. No inversion of control, no event bus, no middleware. The consumer (Alpha-App, Solitude, etc.) orchestrates; the SDK provides the pieces.

## Build & Test Commands

```bash
# Install dependencies (uses uv, not pip)
uv sync

# Run all unit tests
uv run pytest

# Run a single test file
uv run pytest tests/test_proxy.py

# Run a single test by name
uv run pytest tests/test_proxy.py -k "test_name"

# Run integration tests (spawns live claude, costs money)
uv run pytest -m integration

# Skip integration tests (default: they're excluded unless explicitly selected)
uv run pytest -m "not integration"
```

No linter or formatter is configured. No build step — it's a pure Python package built with hatchling.

## Architecture

### The Claude Class (`claude.py`)

The only stateful object. Wraps the `claude` binary subprocess with four I/O channels:

1. **stdin** — JSON messages in (user messages, init handshake, MCP control responses)
2. **stdout** — JSON events out (assistant messages, results, control requests)
3. **stderr** — drained in background, not parsed
4. **HTTP** — localhost reverse proxy (`proxy.py`) intercepts API traffic for compact rewriting and token/usage sniffing

Lifecycle: `Claude()` → `start(session_id?)` → `send(content_blocks)` / `events()` → `stop()`

The proxy is private to Claude — lifecycle 1:1 with the subprocess. No other module imports it.

### Compact Proxy (`proxy.py`)

Sits between claude and Anthropic's API. Three jobs:
- **Rewrite compact requests** — replaces claude's default summarizer identity/instructions/continuation with identity-preserving versions via `CompactConfig`
- **Sniff SSE usage** — extracts input/output/cache tokens from the streaming response (zero extra API calls)
- **Sniff quota headers** — extracts 5h/7d rate limit utilization from response headers

Compact detection uses pinned string signatures from a specific Claude Code version. These break intentionally on upgrade — fix on upgrade.

The rewrite functions (`rewrite_compact`, `_replace_system_prompt`, etc.) are pure — dict in, bool out. Fully unit-testable without running a server.

### System Prompt (`system_prompt.py`)

Assembles identity documents from `$JE_NE_SAIS_QUOI` directory into a flat string for `--system-prompt`. Five pieces concatenated: soul doc, bill of rights, here, capsules, letter. Only soul is required; the rest are optional or TODO.

### MCP Dispatch

In-process FastMCP tool handlers, not external MCP server processes. Claude routes `control_request` messages with subtype `mcp_message` to our handlers; we return `control_response` with the result. Dict in, dict out.

Servers are passed at construction: `Claude(mcp_servers={"cortex": server})`. They get registered as `type: "sdk"` in the merged MCP config.

### Memory System (`memories/`)

Direct Postgres access via asyncpg, embeddings via Ollama:
- `db.py` — low-level Postgres operations, hybrid search (pgvector semantic + tsvector full-text)
- `cortex.py` — high-level API: store, search, recent, get, forget
- `embeddings.py` — embedding generation via configured Ollama model
- `images.py` — image handling for memory storage

### Tools (`tools/`)

- `tools/cortex.py` — MCP tool server wrapping the memory system (store, search, recent, get)

## Key Design Decisions

- **Model is not a constructor default for production** — Alpha pins the model in the consumer. The SDK accepts it as a parameter but the identity's model is an SDK-level decision, not per-call.
- **`JE_NE_SAIS_QUOI`** env var points to the identity directory (soul doc, bill of rights, compact identity). Convention over configuration.
- **Env vars for everything** — no config files. `DATABASE_URL`, `EMBEDDING_MODEL`, `OLLAMA_URL`, `ANTHROPIC_API_KEY`.
- **pytest-asyncio with `asyncio_mode = "auto"`** — async tests just work, no `@pytest.mark.asyncio` needed.
- **Integration tests are marked** — `@pytest.mark.integration` for tests that spawn live claude or need a database. They cost money.
- **`ALPHA_SDK_CAPTURE_REQUESTS=1`** — debug mode that dumps raw proxy requests to `tests/captures/` for inspection.
