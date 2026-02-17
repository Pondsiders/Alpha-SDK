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
- **Pulse** — still schedules Routines
- **Gemma 3 12B on Primer** — recall query extraction and memorables suggestion

## Deployment Model

**Duckpond** installs the SDK as an editable local package (`pip install -e`). It picks up changes immediately — no restart needed for Python changes, restart Duckpond for structural changes. This is the live tinkering environment.

**Routines, Solitude, and other consumers** pin to `alpha_sdk @ git+https://github.com/Pondsiders/Alpha-SDK@main`. They only see code that has been merged to `main` and pushed to GitHub. This is deliberate safety: tinkering on the SDK during the day cannot break Solitude's nighttime breathing.

**The workflow:** tinker on the `tinkering` branch → test live via Duckpond → when happy, merge to `main` and push → other consumers get the changes on next install/restart.

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

### Basic Usage

```python
from alpha_sdk import AlphaClient

# Create once at application startup
client = AlphaClient(
    cwd="/Pondside",
    allowed_tools=[...],
    mcp_servers={...},
)
await client.connect()

# Use for multiple conversations
await client.query(prompt, session_id="abc123")  # Resume existing
async for event in client.stream():
    yield event

await client.query(other_prompt, session_id="xyz789")  # Different session
async for event in client.stream():
    yield event

await client.query(new_prompt)  # New session (no session_id)
async for event in client.stream():
    yield event
print(client.session_id)  # Capture the new session ID

# Cleanup when done
await client.disconnect()
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

### Context Manager (for one-shot use)

```python
async with AlphaClient(cwd="/Pondside") as client:
    await client.query(prompt, session_id=session_id)
    async for event in client.stream():
        yield event
```

## How It Works

### Long-Lived Client & Session Switching

The SDK client is expensive to create (~4 seconds). AlphaClient handles this by:

1. Creating the SDK client once at `connect()`
2. Tracking the current session ID
3. When `query(session_id=X)` is called:
   - If X matches current session → reuse client
   - If X differs → recreate SDK client with new `resume` option
   - If X is None → recreate for new session, capture resulting ID

This means Duckpond can be a dumb pass-through: whatever session_id the frontend sends, AlphaClient figures out whether to reuse or recreate.

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
- Parallel: embedding search + OLMo query extraction
- Deduplicated against session's seen-cache in Redis
- Injected as content blocks (not system prompt)

**After the turn:**
- `suggest()` runs (fire-and-forget)
- OLMo extracts memorable moments
- Results buffer in Redis for potential storage

**On `cortex store`:**
- Memory saved to Postgres with embedding
- Redis buffer cleared

## What Duckpond Becomes

```python
# routes/chat.py - dramatically simplified

@app.post("/api/chat")
async def chat(request: ChatRequest):
    async def generate():
        async with AlphaClient(
            session_id=request.session_id,
            allowed_tools=ALLOWED_TOOLS,
            mcp_servers={"cortex": cortex_server},
        ) as client:
            await client.query(request.content)

            async for event in client.stream():
                yield f"data: {event.json()}\n\n"

            yield f"data: {json.dumps({'type': 'session-id', 'id': client.session_id})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

## What Routines Becomes

```python
# harness.py - also dramatically simplified

async def run_routine(routine: Routine) -> str:
    ctx = RoutineContext(now=pendulum.now("America/Los_Angeles"))
    prompt = routine.build_prompt(ctx)

    async with AlphaClient(
        session_id=routine.get_session_id(),
        fork_from=routine.fork_from_key,
        allowed_tools=routine.get_allowed_tools(),
    ) as client:
        await client.query(prompt)

        output = []
        async for event in client.stream():
            if hasattr(event, 'text'):
                output.append(event.text)

        return routine.handle_output("".join(output), ctx)
```

## Migration Path

1. **Create `alpha_sdk` package** in Basement
2. **Port memory code** from Duckpond's `memories/`
3. **Port system prompt assembly** from Loom's `AlphaPattern`
4. **Build the proxy** (`proxy.py`)
5. **Build the client wrapper** (`client.py`)
6. **Update Duckpond** to use `alpha_sdk`
7. **Update Routines** to use `alpha_sdk`
8. **Decommission** Deliverator, Loom, Argonath

Steps 1-5 can happen without touching Duckpond. Steps 6-7 are the switchover. Step 8 is cleanup.

## What Dies

- **Deliverator** — no more header promotion needed
- **The Loom as a service** — becomes library code
- **Argonath as a service** — becomes library code
- **The hooks** — no longer needed, we control everything upstream
- **The metadata envelope** — no in-band signaling needed
- **Claude Code as a UI** — we use Duckpond exclusively now

## Open Questions

- **Cortex as library vs service?** Currently HTTP. Could become a library too (just Postgres operations). Decided: separate package, dependency of `alpha_sdk`.
- **OLMo location?** Currently on Primer. Stays there for now (GPU).

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

**In production.** Duckpond, Routines, and Solitude all run on alpha_sdk. The tinkering never stops, but the foundation is solid.
