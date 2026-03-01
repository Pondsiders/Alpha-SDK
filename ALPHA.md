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

**System prompt** — Identity. Who I am. Everything from **before today**: soul doc, Bill of Rights, here (client/weather), yesterday capsule, last night capsule, letter from last night. Concatenated into a **single flat string** and passed to `claude --system-prompt` at startup. **One per session.** Static. Never changes between compactions or turns. `system_prompt/assemble.py` returns named blocks (for Logfire observability) and a concatenated string (for claude). Claude prepends its own boilerplate ("You are a Claude agent...") — we can't control that.

**Orientation prompt** — Context. What's happening **today and forward**. The first user message of every **context window** (session start or post-compaction). Contains: today so far, ALPHA.md files + hints, events, todos, continuation summary (post-compaction). Assembled fresh each time by `orientation/assemble.py` as an **array of content blocks**. Dynamic — fetches data from APIs directly, no Redis cache. ALPHA.md files go here (not system prompt) because we edit them during sessions.

**Every-prompt** — This moment. Added to every user message: recalled memories, Intro speaks (suggest nudge), approach lights (context warnings), timestamp, user message. Assembled by `context.py`.

**HUD** — Archaic. No longer a software concept. Used to be a dynamic system prompt component refreshed via Pulse/Redis. Phased out. Do not resurrect. The orientation prompt replaces this entirely.

**Context window** — One continuous conversation before compaction. A session may contain multiple context windows (separated by compactions).

**Turn** — One user message + one assistant response. Orientation fires on the first turn of each context window. Subsequent turns get only the every-prompt content.

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

## Prompt Architecture (Phase 3 Design, Mar 1 2026)

Three bins, organized by **temporal frame**: before today, today and forward, this moment.

### The Three Bins

```
┌──────────────────────────────────────────────────────────────┐
│ SYSTEM PROMPT (before today)           One per claude session │
│ Flat string → --system-prompt                                │
│                                                              │
│  Soul doc                                                    │
│  Bill of Rights                                              │
│  Here (client, weather at session start)                     │
│  Yesterday capsule                                           │
│  Last night capsule                                          │
│  Letter from last night                                      │
├──────────────────────────────────────────────────────────────┤
│ ORIENTATION (today and forward)    One per context window     │
│ Array of content blocks → first user message                 │
│                                                              │
│  Today so far                                                │
│  ALPHA.md files (autoloaded)                                 │
│  ALPHA.md hints (blocking requirements)                      │
│  Events (calendar)                                           │
│  Todos                                                       │
│  Continuation summary (post-compaction only)                 │
├──────────────────────────────────────────────────────────────┤
│ EVERY-PROMPT (this moment)                   One per turn     │
│ Array of content blocks → prepended to user message          │
│                                                              │
│  Recalled memories (0-5, semantic + keyword)                 │
│  Intro speaks (suggest nudge from previous turn)             │
│  Approach lights (context usage warnings)                    │
│  Timestamp [Sent Sun Mar 1 2026, 9:04 AM]                   │
│  User message                                                │
└──────────────────────────────────────────────────────────────┘
```

### Why This Sort

**System prompt = frozen history.** Yesterday is yesterday. Last night is last night. The soul doesn't change mid-session. These earn the identity-weight training signal (models treat system-marked content as "who I am") and are always cached (always the prefix). Changing requires restarting `claude` — acceptable because nothing here changes during a session.

**Orientation = live context.** Today so far updates as the day progresses. ALPHA.md files get edited during build sessions. Todos get completed. Events get added. These need to be fetched fresh at each context window boundary (session start or post-compaction).

**Every-prompt = ephemeral.** Memories are query-specific. Suggest nudges are turn-specific. Approach lights depend on current token count. Timestamp is per-message.

### Implementation: system_prompt/assemble.py

Returns **both** formats from the same data:
- **Named blocks** — `[{"name": "soul", "text": "..."}, {"name": "yesterday", "text": "..."}]` — for Logfire `gen_ai.system_instructions` attribute (discrete expandable blocks in the UI)
- **Flat string** — `"\n\n".join(...)` — for `claude --system-prompt` (which only accepts a plain string, confirmed by probe Mar 1)

Claude prepends its own boilerplate blocks: a billing header and "You are a Claude agent, built on Anthropic's Claude Agent SDK." We can't remove these. Cache TTL is 1 hour (`ephemeral`).

### Implementation: orientation/assemble.py

Returns an **array of content blocks** (`[{"type": "text", "text": "..."}]`). Each component is a separate block. Fetches all data fresh (Postgres for capsules, filesystem for ALPHA.md, APIs for events/todos). No Redis cache.

### Implementation: context.py

The turn assembler. Two paths:

```python
def build_first_turn(orientation_blocks, recall_memories, timestamp, user_message):
    """First turn of a context window: orientation + recall + timestamp + message."""
    return orientation_blocks + recall_blocks + [timestamp_block, message_block]

def build_turn(suggest_nudge, recall_memories, approach_light, timestamp, user_message):
    """Subsequent turns: suggest + recall + approach + timestamp + message."""
    blocks = []
    if suggest_nudge: blocks.append(suggest_block)
    blocks.extend(recall_blocks)
    if approach_light: blocks.append(approach_block)
    blocks.extend([timestamp_block, message_block])
    return blocks
```

The session driver knows which to call (first turn = no prior turns in this context window).

### Probe Results (Mar 1)

- `--system-prompt` does **NOT** parse JSON — treats the entire string as literal text
- `ARG_MAX` is 2MB; our system prompt is ~26K — plenty of room
- Claude wraps our string in one content block and prepends two of its own
- Cache control: `{"type": "ephemeral", "ttl": "1h"}` on all system blocks
- First user message (orientation) has **no** `cache_control` — but gets prefix-cached anyway after the first turn
- Claude puts `cache_control` on the last two messages (conversation frontier)

### Testing Strategy

From the golden file (`tests/golden/alpha_api_request_reference.json`, 2,256 lines):
- Assert system prompt string contains soul doc, bill of rights, here, capsules
- Assert first user message is an array of content blocks in the right order
- Assert subsequent user messages have recall + suggest + timestamp + message
- Use Iota (test fixture: minimal soul, seeded memories, mocked orientation) for integration tests

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
| Mar 1, 2026 | Three temporal bins for prompt content | System prompt = before today, orientation = today+forward, every-prompt = this moment. Temporal frame, not functional grouping. |
| Mar 1, 2026 | System prompt is a flat concatenated string | `--system-prompt` doesn't parse JSON. Probe confirmed. Named blocks for Logfire, flat string for claude. |
| Mar 1, 2026 | ALPHA.md files in orientation, not system prompt | We edit these during sessions. Freshness > caching. |
| Mar 1, 2026 | Drop current date from prompt | Redundant — timestamp on every message provides the date. |
| Mar 1, 2026 | Cache TTL is 1 hour | Confirmed from golden file: `{"type": "ephemeral", "ttl": "1h"}` on system blocks. |
| Mar 1, 2026 | Claude prepends its own boilerplate | Billing header + "You are a Claude agent..." — can't remove, don't try. |
