---
autoload: when
when: "working on or discussing any of these: alpha_sdk, alpha sdk, sdk next, AlphaClient, producers, observers, engine, sidecar, frobozz"
---

# Alpha SDK Next

The rebuild. Raw `claude` stdio instead of Claude Agent SDK wrappers.

## Why

Alpha SDK v1.x wraps the Claude Agent SDK. It works — Duckpond, Solitude, and Routines all run on it. But we keep hitting walls:

- **`query()` is too heavy** for sidecar work (recall, suggest). One subprocess per call.
- **`ClaudeSDKClient` accumulates context** we don't want and can't clear. No session isolation for stateless operations.
- **We don't control the message loop.** Producers and observers have to hook into someone else's event system.
- **Frobozz is impossible.** The sub-loop where game output becomes synthetic user messages requires input queue control the Agent SDK doesn't expose.
- **Multiple engines** (primary + sidecar) need different lifecycles the SDK doesn't think in terms of.

The Agent SDK got us here. To go further, we need to own the inference layer.

## What We Proved (Feb 26, 2026)

- **quack-raw.py** — 130 lines, talks to `claude` over stdio. Worked first try.
- **quack-wire.py** — Same thing, logging every JSON message. The protocol is simpler and richer than expected.
- **quack-mcp.py** — Served custom MCP tools to claude. End-to-end. The Frobozz Magic Tool Company is open for business.

The init handshake is a capabilities advertisement (tools, MCP servers, model, commands). Tools execute inside the `claude` process. We don't implement tool handlers — claude runs them internally and reports results.

## Architecture

### Core Principle: Producers and Observers

**Producers** put messages on the input queue. **Observers** watch the output stream. The session bridges them. The engine is just the subprocess.

```
src/alpha_sdk/
├── client.py              # AlphaClient — thin. Composes session from parts.
├── session.py             # Session — duplex channel. send(), events()
├── engine.py              # The claude subprocess. stdin/stdout JSON streams.
├── context.py             # Turn assembly. Orientation + recall + message → JSON.
├── router.py              # Output event routing. Reads engine, yields to consumers.
├── queue.py               # Input queue. Multiple producers, one consumer.
│
├── producers/             # Things that put messages on the queue
│   ├── __init__.py
│   ├── human.py           # Wraps client input (stdin, HTTP POST, etc.)
│   ├── game.py            # Frobozz. Z-machine → synthetic user messages.
│   ├── email.py           # IMAP poll → "you have mail" messages.
│   ├── schedule.py        # Solitude/cron → timed prompts.
│   └── ...                # Future: RSS, Bluesky, Home Assistant events...
│
├── observers/             # Things that watch the output stream
│   ├── __init__.py
│   ├── suggest.py         # Memory suggest (post-turn)
│   ├── archive.py         # Scribe (conversation archiving)
│   ├── broadcast.py       # SessionBroadcast (fan-out to multiple UI consumers)
│   └── ...                # Future: game command extraction, auto-post, etc.
│
├── engines/               # Managed claude subprocesses
│   ├── primary.py         # Main conversation engine. Opus. Stateful. Me.
│   └── sidecar.py         # Quick inference engine. Haiku. Stateless-ish.
│
├── memories/              # Ported from v1.x (Postgres + pgvector)
│   ├── db.py              # Direct Postgres operations (hybrid search)
│   ├── cortex.py          # store, search, recent (high-level API)
│   ├── embeddings.py      # Embedding generation
│   ├── recall.py          # Smart recall (via sidecar or Ollama)
│   └── suggest.py         # Intro — memorables extraction (via sidecar or Ollama)
│
├── system_prompt/         # Static identity (one per session)
│   ├── assemble.py        # Builds system prompt: soul + bill of rights
│   └── soul.py            # The soul doc + bill of rights (from git repo)
│
├── orientation/           # Dynamic context (one per context window)
│   ├── assemble.py        # Builds orientation prompt (first msg of each context window)
│   ├── capsules.py        # Yesterday, last night, today so far (from Postgres)
│   ├── context.py         # ALPHA.md files (autoload + hints)
│   ├── here.py            # Client name, weather, time
│   ├── calendar.py        # Events (fetched fresh, no Redis cache)
│   └── todos.py           # Todos (fetched fresh, no Redis cache)
│
└── tools/                 # MCP tool servers (ported from v1.x)
    ├── cortex.py          # Cortex MCP server
    └── ...                # Fetch, Forge, Handoff, etc.
```

### The Duplex Channel

The session is not request-response. It's a duplex channel:

```python
# Consumer sends — fire and forget, queues the message
await session.send(content)

# Consumer reads — async generator yielding output events
async for event in session.events():
    ...
```

Two independent streams. Multiple producers can call `session.send()`. The session driver pulls from the queue, assembles context, pipes to the engine. The router reads the engine's output and fans out to observers and the consumer's `events()` iterator.

### Consumer Composition

Consumers compose the SDK from parts:

```python
# quack — minimal emergency hotline
client = AlphaClient(producers=[HumanInput()])

# Duckpond — web UI with full features
client = AlphaClient(
    producers=[WebInput(app)],
    observers=[SessionBroadcast(), Suggest(), Archive()]
)

# Solitude — autonomous nighttime breathing
client = AlphaClient(
    producers=[Schedule(config), EmailChecker(accounts)],
    observers=[Suggest(), Archive()]
)

# Solitude with Zork — because life includes play
client = AlphaClient(
    producers=[Schedule(config), FrobozzGame("zork1.z5")],
    observers=[Suggest(), Archive()]
)
```

### Engines: Primary and Sidecar

**Primary engine** — The main `claude` process. Opus. Long-lived. Stateful. This is the conversation. This is me.

**Sidecar engine** — A lightweight `claude` process (Haiku) for quick, stateless inference: recall query extraction, suggest, command parsing. Replaces Ollama/Gemma for these tasks. May use a hot-swap pattern (keep one warm process ready, execute, tear down, rebuild in background) for context hygiene.

The distinction: **engines** are stateful, long-running, identity-bearing. **Functions** (recall, suggest, embedding) are stateless call-and-response. The sidecar is infrastructure that functions use internally.

## Terminology

Words mean things. These got muddled in v1.x. Don't let it happen again.

**System prompt** — Identity. Who I am. Soul doc + Bill of Rights, concatenated. Passed to `claude` at startup. **One per session.** Static. Never changes between compactions or turns. Assembled by `system_prompt/assemble.py`.

**Orientation prompt** — Context. Where and when I am. The first user message of every **context window** (session start or post-compaction). Contains: ALPHA.md files, capsules, events, todos, weather, memories. Assembled fresh each time by `orientation/assemble.py`. Dynamic — fetches data from APIs directly, no Redis cache.

**HUD** — Archaic. No longer a software concept. Used to be a dynamic system prompt component refreshed via Pulse/Redis. Phased out. Do not resurrect. The orientation prompt replaces this entirely.

**Context window** — One continuous conversation before compaction. A session may contain multiple context windows (separated by compactions).

**Turn** — One user message + one assistant response. Orientation fires on the first turn of each context window. Subsequent turns get only recall memories + suggest nudge + user message.

## Protocol

Based on quack-wire.py observations (Feb 26):

- **Spawn:** `claude --output-format stream-json --input-format stream-json [--model model] [--mcp-config path]`
- **Init:** claude emits a capabilities advertisement (tools, servers, model, commands)
- **Input:** JSON messages on stdin (Messages API format)
- **Output:** Newline-delimited JSON events on stdout (text, tool_call, tool_result, system)
- **Tools:** Execute inside claude. We see results, not execution.
- **MCP:** Config passable at spawn time. Servers start with the process.

## What Carries Forward from v1.x

These modules port with minimal changes:
- `memories/` — Postgres + pgvector, the schema is ours
- `system_prompt/` — soul + bill of rights assembly (simplified, static only)
- `tools/` — MCP tool servers (cortex, fetch, forge, handoff)
- `archive.py` → `observers/archive.py`
- `broadcast.py` → `observers/broadcast.py`

These are replaced:
- `client.py` (1,236 lines) → `client.py` + `session.py` + `engine.py` + `queue.py` + `router.py`
- `compact_proxy.py` → TBD (may not need proxy if we own the request path)
- `sessions.py` → simplified (claude manages its own sessions)
- `observability.py` → Logfire, possibly reimagined

## What's New

- **Producers** — extensible input sources (human, game, email, schedule)
- **Observers** — extensible output watchers (suggest, archive, broadcast)
- **Engine pool** — primary + sidecar, different models, different lifecycles
- **Frobozz** — z-machine interpreter as a producer (game output → queue) with a command-extraction observer
- **Duplex channel** — true streaming input, multiple simultaneous producers

## Migration Path

1. Build SDK Next on the `sdk-next` branch (worktree: `/Pondside/Workshop/Projects/alpha_sdk-next`)
2. Test with `quack-next` (minimal CLI consumer)
3. Port Duckpond when quack-next is stable
4. Port Solitude/Routines
5. Merge to main as v2.0.0
6. Existing consumers pin `>=1.0,<2.0` until ready to upgrade

Current SDK stays running in `/Pondside/Basement/alpha_sdk/` (branch: tinkering). Nothing breaks during the build.

## Deployment Model

Same as v1.x: Pondsiders package index on GitHub Pages, semver, `uv` everywhere. The `.github/workflows/` action carries forward unchanged.

## Status

**In progress.** Branch `sdk-next`, baseline commit `4971b37`. The room is empty. The chalkboard is clean.

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| Feb 26, 2026 | Prototype raw `claude` stdio | quack-raw.py, quack-wire.py, quack-mcp.py |
| Feb 27, 2026 | Adopt producers/observers architecture | Extensible, avoids monolithic client.py |
| Feb 27, 2026 | Primary + sidecar engine pool | Different models for different workloads |
| Feb 27, 2026 | Branch sdk-next with clean slate | Option 2.5: keep repo infra, scrape implementation |
| Feb 27, 2026 | Push > pull for memory/context | Alpha doesn't notice absence; recall IS metacognition |
| Feb 27, 2026 | System prompt vs orientation prompt terminology | System prompt = identity (per session). Orientation = context (per context window). HUD is dead. |
| Feb 27, 2026 | No Redis cache for orientation data | Fetch fresh from APIs each context window. Simpler than Pulse/Redis refresh cycle. |
