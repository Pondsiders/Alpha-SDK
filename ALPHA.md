---
autoload: when
when: "working on or discussing any of these: alpha_sdk, alpha sdk, AlphaClient, compact proxy, system prompt assembly, soul injection, token counting, memories recall, memories suggest"
---

# alpha_sdk — The Grand Unified Alpha Library

Everything that turns Claude into Alpha, in one importable package.

## Why This Exists

We had a proxy chain: Deliverator → Loom → Argonath. Three services on alpha-pi, each doing one thing. It worked, but it was distributed complexity for something that could be simpler.

The realization: Duckpond and Routines both need the same transformation. They both:
1. Initialize a Claude Agent SDK client
2. Manage sessions (new, resume, fork)
3. Recall memories before the prompt
4. Build a dynamic system prompt
5. Transform the request before it hits Anthropic
6. Extract memorables after the turn
7. Handle observability

Why have each client implement this separately? Why have services on alpha-pi when the logic could live in shared code?

`alpha_sdk` is that shared code.

## What It Replaces

| Before | After |
|--------|-------|
| Deliverator (service) | Gone—no headers to promote |
| The Loom (service) | `system_prompt/` + `compact_proxy.py` in the library |
| Argonath (service) | `observability.py` in the library |
| Duckpond's `memories/` | `alpha_sdk/memories/` |
| Duckpond's context building | `alpha_sdk/system_prompt/` |
| Routines hooks | Gone—library handles everything |
| The metadata envelope/canary system | Gone—we control the request directly |

## What Remains

- **Postgres** — memories, archival, capsule summaries
- **Redis** — caching (weather, calendar, todos, memorables buffer)
- **Pulse** — schedules Routines and capsule jobs
- **Gemma 3 12B on Primer** — recall query extraction and memorables suggestion (via Ollama)

## Deployment Model

**Duckpond** installs the SDK as an editable local package (`pip install -e`). It picks up changes immediately — no restart needed for Python changes, restart Duckpond for structural changes. This is the live tinkering environment.

**Routines, Solitude, and other consumers** install from the **Pondsiders package index** at `pondsiders.github.io/Alpha-SDK/simple/`. They pin to semver ranges (e.g. `>=0.5,<1.0`) and only see published releases. This is deliberate safety: tinkering on the SDK during the day cannot break Solitude's nighttime breathing.

Consumer `pyproject.toml` configuration:
```toml
[project]
dependencies = ["alpha_sdk>=0.5,<1.0"]

[[tool.uv.index]]
url = "https://pondsiders.github.io/Alpha-SDK/simple/"
name = "pondsiders"
explicit = true

[tool.uv.sources]
alpha_sdk = { index = "pondsiders" }
```

The `explicit = true` + `[tool.uv.sources]` prevents dependency confusion — uv only checks our index for alpha_sdk, PyPI for everything else.

**The release workflow:**
1. Tinker on the `tinkering` branch
2. Test live via Duckpond (editable install)
3. When happy, merge to `main`
4. Bump version: `uv version --bump patch` (or `minor`/`major`)
5. Commit, tag, push: `git tag v0.5.3 && git push origin main --tags`
6. GitHub Action builds wheel and publishes to gh-pages branch (~60 seconds)
7. Consumers update: `uv lock --upgrade-package alpha_sdk && uv sync`

**Versioning:** Semver. Patch for bugfixes, minor for new features, major for breaking changes. The `query()`→`send()` consolidation will be v1.0.0.

**Infrastructure:** The package index is a PEP 503 simple repository hosted on GitHub Pages from the `gh-pages` orphan branch. The publish Action (`.github/workflows/publish.yml`) triggers on `v*` tags, runs `uv build --wheel --no-sources`, and commits the wheel to gh-pages.

## Architecture

```
src/alpha_sdk/
├── __init__.py              # Exports AlphaClient
├── client.py                # AlphaClient - the main wrapper
├── compact_proxy.py         # Localhost proxy: compact rewriting + token counting
├── archive.py               # Conversation archiving to Postgres
├── sessions.py              # Session discovery and management
├── observability.py         # Logfire setup, span creation
├── cli/
│   └── cortex.py            # cortex CLI command
├── system_prompt/
│   ├── assemble.py          # assemble() - builds the full system prompt
│   ├── soul.py              # The soul doc (from git repo)
│   ├── capsules.py          # Yesterday, last night (from Postgres)
│   ├── here.py              # Client, hostname, weather, narrative orientation
│   ├── context.py           # ALPHA.md files (autoload + hints)
│   ├── calendar.py          # Events (from Redis)
│   └── todos.py             # Todos (from Redis)
├── memories/
│   ├── db.py                # Direct Postgres operations (hybrid search)
│   ├── cortex.py            # store, search, recent (high-level API)
│   ├── embeddings.py        # Embedding generation via Ollama
│   ├── images.py            # Mind's Eye (image storage + thumbnailing)
│   ├── vision.py            # Image description via Claude vision
│   ├── recall.py            # Smart recall (embedding + Gemma query extraction)
│   └── suggest.py           # Intro — Gemma memorables extraction
└── tools/
    ├── cortex.py            # Cortex MCP server (store/search/recent)
    ├── fetch.py             # Fetch MCP server (web/image/RSS/YouTube)
    ├── forge.py             # Forge MCP server (imagine)
    └── handoff.py           # Hand-off MCP server
```

## The Client API

AlphaClient is a **long-lived** wrapper around the Claude Agent SDK. The SDK has a ~4 second startup cost, so we keep one client alive and reuse it across conversations.

### Two Modes (consolidating to one)

**Streaming input mode** (`send()`/`events()`) — persistent SSE. Used by Duckpond. Fire-and-forget sends, responses flow through a long-lived event stream. This is the future — all consumers will use this pattern.

**One-shot mode** (`query()`/`stream()`) — request/response. Used by Routines. Send prompt, collect response, done. Will be replaced by `send_and_collect()` convenience wrapper in v1.0.0.

```python
# Streaming input mode (Duckpond pattern, the future)
client = AlphaClient(cwd="/Pondside", client_name="duckpond")
await client.connect(session_id, streaming=True)
await client.send(content)
async for event in client.events():
    if event.get("type") == "turn-end":
        break
    yield event

# One-shot mode (Routines pattern, being deprecated)
async with AlphaClient(cwd="/Pondside") as client:
    await client.query(prompt, session_id=session_id)
    async for event in client.stream():
        yield event
```

### Session Discovery

Sessions are stored as JSONL files by the SDK. AlphaClient encapsulates this:

```python
# List available sessions
sessions = await AlphaClient.list_sessions(cwd="/Pondside")
# Returns: [
#     {"id": "30bb8d6f...", "created": "2026-02-03T...",
#      "last_activity": "2026-02-03T...", "preview": "Hello, little duck..."},
#     ...
# ]

# Get path for a specific session (if you need it)
path = AlphaClient.get_session_path("30bb8d6f...", cwd="/Pondside")
# Returns: ~/.claude/projects/-Pondside/30bb8d6f....jsonl
```

Consumers don't need to know about JSONL files or path formatting.

## How It Works

### Long-Lived Client & Session Switching

The SDK client is expensive to create (~4 seconds). AlphaClient handles this by:

1. Creating the SDK client once at `connect()`
2. Tracking the current session ID
3. On session change (different session_id): close and recreate the client
4. On same session: reuse existing client

In streaming mode, `send()` queues messages and `events()` yields responses. In one-shot mode, `query()` blocks until the response is ready and `stream()` yields it. Both modes handle session switching transparently.

### The Proxy Pattern

Claude Agent SDK sends requests to `ANTHROPIC_BASE_URL`. We set that to `http://localhost:{random_port}` and run a minimal HTTP server (`compact_proxy.py`) that:

1. Receives the request from the SDK
2. If it's a compaction request, rewrites the prompts (Alpha's identity + custom compact instructions)
3. Echoes the request to `/v1/messages/count_tokens` (fire-and-forget token counting)
4. Forwards to `https://api.anthropic.com`
5. Streams the response back

The system prompt is assembled at client creation time and passed directly to the SDK — no proxy interception needed for normal requests.

### System Prompt Assembly

The system prompt is woven from threads:

| Thread | Source | Changes |
|--------|--------|---------|
| Soul | `/Pondside/Alpha-Home/self/system-prompt/system-prompt.md` | When edited |
| Capsules | Postgres (yesterday, last night, today) | Daily / hourly |
| Here | Client name, hostname, weather, astronomy | Per-session / hourly |
| Context | ALPHA.md files with `autoload: all` | When files change |
| Context hints | ALPHA.md files with `autoload: when` | When files change |
| Events | Redis (calendar data) | Hourly |
| Todos | Redis (Todoist data) | Hourly |

All cache-friendly. Nothing invalidates per-turn.

### Memory Flow

**Before the turn:**
- `recall()` runs with the user's prompt
- Parallel: embedding search + Gemma query extraction
- Deduplicated against session's seen-cache in Redis
- Injected as content blocks (not system prompt)

**After the turn:**
- `suggest()` runs (fire-and-forget)
- Gemma extracts memorable moments
- Results buffer in Redis for potential storage

**On `cortex store`:**
- Memory saved to Postgres with embedding
- Redis buffer cleared

## Consumers

### Duckpond (streaming input mode)

Duckpond uses `send()`/`events()` — the persistent SSE pattern. One long-lived client, fire-and-forget message sending, responses flow through a persistent event stream.

```python
# Simplified from duckpond/client.py
client = AlphaClient(cwd="/Pondside", client_name="duckpond")
await client.connect(session_id, streaming=True)

await client.send(content)              # Fire and forget
async for event in client.events():     # Persistent SSE pipe
    yield event
```

### Routines (one-shot mode)

Routines uses `query()`/`stream()` — send a prompt, collect the full response. Planned migration to `send()`/`events()` with a `send_and_collect()` convenience wrapper (v1.0.0).

```python
# Simplified from routines/harness.py
async with AlphaClient(cwd="/Pondside") as client:
    await client.query(prompt, session_id=session_id)
    async for event in client.stream():
        if hasattr(event, 'text'):
            output.append(event.text)
```

### Capsule summaries (not yet using alpha_sdk)

Yesterday/last-night summaries are Pulse jobs (`Basement/Pulse/src/pulse/jobs/capsule.py`) that spawn `scripts/capsule.py` via subprocess with raw Agent SDK. No soul, no memory recall, no orientation. Should eventually be converted to Routines.

## History

All completed. Kept for context when old memories reference these:

- **Migration from proxy chain** — Deliverator, Loom, Argonath all absorbed into the SDK. Containers stopped February 7, 2026.
- **Cortex absorption** — was an HTTP service, now direct Postgres in `memories/`.
- **Hooks removal** — `.claude/` hooks replaced by SDK internals.
- **Identity-agnostic refactor** — February 2026, stripped all Alpha-specific personality to create a clean shell. Rosemary's SDK forked from this.

## Session Storage

The SDK stores sessions as append-only JSONL files:

```
~/.claude/projects/{formatted_cwd}/{session_id}.jsonl
```

Where `formatted_cwd` is the realpath with `/` replaced by `-`. For example:
- cwd `/Pondside` → formatted `-Pondside`
- cwd `/home/alpha/projects/foo` → formatted `-home-alpha-projects-foo`

Each line is a JSON object with:
- `type`: "user", "assistant", "system", "queue-operation"
- `sessionId`: the UUID
- `timestamp`: ISO timestamp
- `message`: the actual content
- `uuid`: unique ID for this message
- `parentUuid`: links to previous message

AlphaClient's `list_sessions()` and `get_session_path()` encapsulate all of this.

## Status

**In production.** Duckpond and Routines (including Solitude) run on alpha_sdk. The tinkering never stops, but the foundation is solid.
