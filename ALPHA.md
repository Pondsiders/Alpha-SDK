---
autoload: when
when: "working on or discussing any of these: alpha_sdk, alpha sdk, sdk next, AlphaClient, producers, observers, engine, sidecar, frobozz"
---

# Alpha SDK Next

The rebuild. Raw `claude` stdio instead of Claude Agent SDK wrappers.

**The mirepoix principle:** The SDK is the foundation that doesn't taste like anyone. Alpha is the stew. Clyde is the broth. Mr. House, Rosemary, future consumers — all different recipes, same base. The personalization lives *above* the SDK, not inside it.

## Why

Alpha SDK v1.x wraps the Claude Agent SDK. It works — Duckpond, Solitude, and Routines all run on it. But we keep hitting walls:

- **`query()` is too heavy** for sidecar work (recall, suggest). One subprocess per call.
- **`ClaudeSDKClient` accumulates context** we don't want and can't clear. No session isolation for stateless operations.
- **We don't control the message loop.** Producers and observers have to hook into someone else's event system.
- **Frobozz is impossible.** The sub-loop where game output becomes synthetic user messages requires input queue control the Agent SDK doesn't expose.
- **Multiple engines** (primary + sidecar) need different lifecycles the SDK doesn't think in terms of.

The Agent SDK got us here. To go further, we need to talk to `claude` directly.

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
├── engine.py              # The claude subprocess. stdin/stdout/stderr + HTTP proxy.
├── proxy.py               # HTTP channel. Compact rewriting + token counting. Engine-private.
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
│   └── suggest.py         # Memorables extraction (via sidecar or Ollama)
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

### The Four I/O Channels

The claude subprocess has four communication channels:

1. **stdin** — JSON messages in (user messages, init handshake, permission responses)
2. **stdout** — JSON events out (assistant messages, results, system events)
3. **stderr** — Diagnostic output (drained in background, not parsed)
4. **HTTP to Anthropic** — API requests via `ANTHROPIC_BASE_URL` (inference, compaction)

The Engine manages all four. Channels 1-3 are direct subprocess pipes. Channel 4 is intercepted by a localhost HTTP proxy (`proxy.py`) that the Engine starts before spawning claude. The proxy:

- **Rewrites compact requests** — Replaces claude's default summarizer identity, compact instructions, and continuation instruction with Alpha's versions. Three-phase surgical rewrite.
- **Counts tokens** — Fire-and-forget echo to `/v1/messages/count_tokens` on every API request. Tracks high-water mark.
- **Sniffs usage headers** — Extracts `anthropic-ratelimit-unified-{7d,5h}-utilization` from Anthropic's response headers.

The proxy is a private implementation detail of Engine — no other module imports or knows about it.

## What Carries Forward from v1.x

These modules port with minimal changes:
- `memories/` — Postgres + pgvector, the schema is ours
- `system_prompt/` — soul + bill of rights assembly (simplified, static only)
- `tools/` — MCP tool servers (cortex, fetch, forge, handoff)
- `archive.py` → `observers/archive.py`
- `broadcast.py` → `observers/broadcast.py`

These are replaced:
- `client.py` (1,236 lines) → `client.py` + `session.py` + `engine.py` + `queue.py` + `router.py`
- `compact_proxy.py` → `proxy.py` (Engine-private. Compact rewriting + token counting + usage headers.)
- `sessions.py` → simplified (claude manages its own sessions)
- `observability.py` → Logfire, possibly reimagined

## What's New

- **Producers** — extensible input sources (human, game, email, schedule)
- **Observers** — extensible output watchers (suggest, archive, broadcast)
- **Engine pool** — primary + sidecar, different models, different lifecycles
- **Frobozz** — z-machine interpreter as a producer (game output → queue) with a command-extraction observer
- **Duplex channel** — true streaming input, multiple simultaneous producers

## Branching & Versioning

**Version history:** 0.x was the prototype SDK (still running in Basement on `tinkering`). We never published 1.0.0. The rewrite IS the 1.0.

**Branch topology:**
- `main` — the plain SDK. Engine, session, queue, router, producers, observers. Eventually: memory machinery, soul loading, orientation assembly — all disabled-by-default infrastructure. Clyde consumes published releases from here.
- `alpha` (future) — branches off `main`, carries Alpha-specific code: soul doc paths, Alpha-specific observers, Frobozz, Solitude producers. Merge from `main` (not rebase) to pick up infrastructure improvements.
- `tinkering` — legacy v0.x code. Duckpond/Solitude/Routines still run on this in `/Pondside/Basement/alpha_sdk/`. Stays until those consumers port to v1.x.

**Forks for other personalities:**
- Rosemary-SDK: fork of Alpha-SDK `main`. Merges upstream when ready.
- House-SDK, others: same pattern.

**Versioning:** Semver. `1.0.0a1` = first alpha (plain SDK, mirepoix only). Bump alpha versions as infrastructure is added. `1.0.0` = the full SDK with all generic machinery. Alpha-specific releases may use a different scheme TBD.

**Deployment:** Pondsiders package index on GitHub Pages, `uv` everywhere. `.github/workflows/` action publishes wheels on version tags.

## First Consumer: Clyde (Project M.O.O.S.E.)

**Clyde** is the broth test — the simplest possible consumer that validates the mirepoix. A stateless Haiku web app that replaces Jeffery's ChatGPT subscription ($20/month). No soul, no memory, no MCP tools. One producer (web UI), zero observers. Rosemary's frontend, reskinned.

Clyde proves the SDK works before we layer on Alpha-specific complexity. If the broth tastes wrong, the mirepoix is wrong.

### Clyde Architecture (designed Feb 28)

**Frontend:** Copy Rosemary-App's 6 files with string swaps. assistant-ui + Zustand + Tailwind v4. Dark theme with ChatGPT green (`#10a37f`). Image attachment via `SimpleImageAttachmentAdapter` (base64 passthrough — no server-side storage). Drop file upload adapter.

**Backend:** FastAPI serving built Vite static files + SSE streaming endpoint.
- `client.py` — wraps `AlphaClient(model="haiku")`. Simpler than Rosemary's `GreenhouseClient` (no SDK preprocessing).
- `routes/chat.py` — POST /api/chat → translates `AlphaClient.events()` to SSE (`AssistantEvent` → `text-delta`, `ResultEvent` → `session-id` + `done`).
- `routes/sessions.py` — GET /api/sessions from Redis. GET /api/sessions/{id} from JSONL files.
- `main.py` — FastAPI, Logfire, static serving, health check.

**Persistence:** Bind mount to Pondside (`/Pondside/Workshop/Projects/Clyde/data/`).
- `data/claude/` → mounted as `~/.claude` (session JSONL files)
- `data/redis/` → Redis AOF persistence

Inherits Syncthing (3 machines) + Restic (B2) backup automatically. 30-day natural expiry matches claude's session retention.

**Docker:** Multi-stage Dockerfile (node builds Vite → python serves FastAPI). Redis Alpine sidecar with `--appendonly yes`. Port 8780 (or similar).

**What Clyde doesn't have:** Neon, any SDK beyond AlphaClient, nights.py, summaries.py, upload route, context route, system prompt, memory, MCP tools.

**Rosemary files reviewed Feb 28:** ~1,500 lines frontend, ~600 lines backend. Genuinely portable. Only real new code is the backend client wrapper and SSE translation layer.

## Session Design

`engine.start(session_id=None)` — returns None, not a session ID:
- **No session_id:** spawn claude. Session ID is `None` until first turn — arrives on `ResultEvent`.
- **With session_id:** replay JSONL as events with `is_replay=True`, spawn claude with `--resume`.

Wire protocol truth (confirmed Feb 28): claude emits **zero** session identity during init. One `control_response` with model/tools/servers. Session ID first appears on the first turn's `ResultEvent`. Every message after that carries it.

Same pipe for replay and live. The consumer doesn't need two code paths. Events are events — some came from disk (fast), some from the subprocess (real-time). `is_replay` flag on every event lets consumers distinguish if they want to (batch-render history, skip animation) but they don't have to.

## Status

**In progress.** Branch `sdk-next`. Phases 0, 1, 1.5, and 2 complete. Phase 2 (the mirepoix) shipped: queue, router, replay, session, client, producers/human. 166 tests, zero failures. quack-next.py validates end-to-end with Haiku. Phase 2.5 (Clyde) is next.

Granular progress tracking lives in [#22](https://github.com/Pondsiders/Alpha-SDK/issues/22). This doc is the reference architecture; the issue is the punch list.

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
| Feb 27, 2026 | HTTP proxy is Engine-private (Phase 1.5) | Fourth I/O channel (compact rewrite + token counting). No independent lifecycle — born/dies with subprocess. |
| Feb 28, 2026 | Mirepoix reframe — SDK as generic foundation | Build the base that doesn't taste like anyone. Personalization lives above the SDK. |
| Feb 28, 2026 | Clyde as first consumer (Phase 2.5) | Validates the mirepoix before adding Alpha-specific ingredients. Replaces ChatGPT ($20/mo). |
| Feb 28, 2026 | Session replay via JSONL → events pipe | `engine.start(session_id)` replays history with `is_replay=True`. Same pipe, same types. |
| Feb 28, 2026 | `is_replay` flag on Event base class | Metadata is cheap to include and expensive to add later. |
| Feb 28, 2026 | `start()` returns None, not session_id | Wire protocol confirmed: claude emits no session_id during init. It arrives on first turn's ResultEvent. Don't promise what you can't deliver. |
| Feb 28, 2026 | Phase 2 complete — the mirepoix | queue.py, router.py, replay.py, session.py, client.py, producers/human.py. 166 tests, 0 failures. quack-next.py validates end-to-end. |
| Feb 28, 2026 | Version is 1.0.0, not 2.0.0 | We never published 1.0.0. The 0.x series was prototype. The rewrite is the real 1.0. |
| Feb 28, 2026 | Clyde is a separate project, not in-repo | Clyde is a real product (ChatGPT replacement), not a test harness. Pins to published SDK release. |
| Feb 28, 2026 | `main` = plain SDK, `alpha` = personality branch | Optimize for freedom to tinker on Alpha without breaking downstream. Merge from main, not rebase. |
| Feb 28, 2026 | "Thick main" — memory/soul/orientation machinery on `main` | Infrastructure is hippocampus, not personality. Disabled-by-default. Even Clyde has it, just doesn't enable it. |
| Feb 28, 2026 | Forks for other personalities (Rosemary, House) | Fork `main`, merge upstream. No entanglement with Alpha-specific code. |
