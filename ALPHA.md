---
autoload: when
when: "working on or discussing any of these: alpha_sdk, alpha sdk, sdk next, AlphaClient, producers, observers, engine, sidecar, frobozz"
---

# Alpha SDK Next

The rebuild. Raw `claude` stdio instead of Claude Agent SDK wrappers.

**The mirepoix principle:** The SDK is the foundation that doesn't taste like anyone. Alpha is the stew. Clyde is the broth. Mr. House, Rosemary, future consumers — all different recipes, same base. The personalization lives *above* the SDK, not inside it.

## Why

Alpha SDK v0.x wraps the Claude Agent SDK. It works — Duckpond, Solitude, and Routines all run on it. But we keep hitting walls:

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
├── providers/             # Block providers. Drop a file, get a block.
│   ├── soul.py            # generic. BIN=system. Reads soul doc from JE_NE_SAIS_QUOI.
│   ├── here.py            # generic. BIN=system. Client, weather, hostname.
│   ├── recall.py          # generic. BIN=turn. Memory recall (if memory enabled).
│   ├── suggest_nudge.py   # generic. BIN=turn. Intro speaks (if memory enabled).
│   ├── approach.py        # generic. BIN=turn. Context usage warnings.
│   ├── capsules.py        # Alpha-specific. BIN=system. Yesterday, last night (Postgres).
│   ├── letter.py          # Alpha-specific. BIN=system. Letter from last night (Redis).
│   ├── today.py           # Alpha-specific. BIN=orientation. Today so far (Redis).
│   ├── context_files.py   # Alpha-specific. BIN=orientation. ALPHA.md autoload + hints.
│   ├── calendar.py        # Alpha-specific. BIN=orientation. Events (API).
│   └── todos.py           # Alpha-specific. BIN=orientation. Todos (API).
│
├── assemble.py            # The loom. Discovers providers, assembles by bin.
│
└── tools/                 # Built-in MCP tool servers
    ├── cortex.py          # generic. Memory tools (follows memory config)
    ├── fetch.py           # generic. Web fetching
    └── handoff.py         # generic. Context window management
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

## Session Design: Lazy Kernels

Designed March 2, 2026. Adversarial-reviewed by Opus and passed.

### The Metaphor

Each chat has a **kernel** — a `claude` subprocess — that is either running or not. Like a Jupyter notebook: one chat, one kernel, the kernel has state. Kernels start lazily (on first message, not on chat open), run as long as needed, and get reaped after idle timeout.

### Chat IDs vs Session IDs

Two ID systems, clearly separated:

| | Chat ID | Session ID |
|---|---------|------------|
| **Generated by** | Frontend (nanoid) | `claude` subprocess |
| **Format** | Short, URL-friendly (e.g., `k7x9mQ2p`) | UUID (e.g., `469740c6-27b6-4094-874c-6d5aa1dfa58a`) |
| **Scope** | Our thing. URLs, WebSocket, sidebar, Redis. | Claude's thing. `--resume`, JSONL transcripts. |
| **Lifecycle** | Born when user clicks "New Chat" | Born on first `ResultEvent` from claude |

The backend maintains a mapping: `chatId → claudeSessionId`. Claude's UUID is an implementation detail. If we ever switch providers, chat IDs don't change — only the mapping does.

### Kernel Lifecycle

```
User clicks "New Chat"
  → Frontend generates chatId = nanoid()
  → Chat enters PENDING state — sidebar shows it (dimmed/ghostly)
  → At most ONE pending chat exists at a time

User clicks "New Chat" again (pending chat already exists)
  → Guard-flip: button highlights/warns on first click, executes on second
  → Old pending chat is replaced, new one takes its place

User types a message and hits send (pending chat)
  → Chat transitions: PENDING → STARTING → BUSY
  → claude starts fresh (no --resume)
  → Events stream back, light goes green (pulsing)
  → ResultEvent arrives: BUSY → IDLE, light goes amber
  → Session ID from ResultEvent stored in chatId→sessionId mapping
  → Chat is now real — it persists

Error on first send (pending chat)
  → claude barfs, error propagates to browser
  → Chat stays PENDING — it never became real
  → User can retry (same chat) or click New Chat to replace it
  → No orphaned dead chats from failed first messages

User clicks existing chat
  → History replays from JSONL transcript (no subprocess needed)
  → Gray light stays gray — browsing is free

User types a message (existing chat, kernel dead)
  → Kernel starts: DEAD → STARTING → BUSY
  → claude starts with --resume <sessionId>
  → Events stream back normally

No message for 10 minutes (idle timeout)
  → Kernel reaped: IDLE → DEAD, light goes gray
  → Active kernels (BUSY) are never reaped

User switches to another chat and types
  → New kernel starts for that chat
  → Old kernel stays alive (amber) until reaped
  → Two kernels can briefly coexist — visible as two amber dots
```

### State Machine

Five states per chat, transitions must be atomic:

```
                send()          events flowing        ResultEvent
PENDING ──────→ STARTING ──────────→ BUSY ──────────→ IDLE
                  ↑                                    │
  DEAD ───────────┘                                    │
   ↑                                                   │
   └────────────────── reaper (10 min) ────────────────┘
   ↑                                    │
   └──────── crash / unexpected exit ───┘
```

- **PENDING** — Chat ID exists but no session ID. No subprocess has ever been asked to do anything. At most one pending chat at a time. Sidebar: dimmed/ghostly, no indicator light. Not yet born.
- **DEAD** — Had a kernel, now doesn't. Opening the chat replays history from disk. Light: gray.
- **STARTING** — Subprocess spawning. Messages that arrive queue up. Light: amber.
- **IDLE** — Subprocess alive, waiting for input. Light: amber.
- **BUSY** — Subprocess processing a message. Light: green (pulsing).

Messages arriving during STARTING must queue. Only one spawn attempt per chat (prevent double-spawn on rapid sends). The reaper only touches IDLE kernels — BUSY and STARTING are protected.

### Failure Modes

**Subprocess crash:** Monitor process handle. On unexpected exit → transition to DEAD, notify frontend via WebSocket (light goes gray + error indicator), log failure.

**Subprocess hang:** If BUSY for more than N minutes without producing events → force kill, transition to DEAD.

**WebSocket disconnect (laptop lid close):** On reconnect, frontend sends "I last saw message N for chat X." Backend replays from JSONL transcript to catch up. Same replay mechanism as chat-open — one code path.

**Backend restart:** All subprocesses die (use `pdeathsig` so they don't orphan). Everything starts cold. Users see gray lights, type to restart kernels. In-flight responses are lost but transcripts on disk are intact.

### The Reaper

Idle = `ResultEvent` received AND no new user message since. This is precise:
- A subprocess waiting for tool confirmation is NOT idle (it's BUSY, mid-turn).
- A subprocess that just finished responding IS idle.

Timer resets on each `send()`. Uses `time.monotonic()` (immune to clock adjustments). Reaper checks on a configurable interval (e.g., every 60 seconds). Active kernels (BUSY, STARTING) are untouchable.

Safety valve: if total managed subprocesses exceeds a threshold (e.g., 20), force-reap oldest idle ones regardless of timeout.

### Transport: WebSocket

One WebSocket per browser tab. Replaces per-POST SSE.

```
Browser ←── WebSocket ──→ Backend
              │                │
    Messages up (JSON):        Events down (JSON):
    {chatId, content}          {chatId, type, data}
                               type: text-delta, thinking-delta,
                                     tool-call, session-id,
                                     history-replay, error, done
```

**Why WebSocket over SSE:**
- Bidirectional. User can send while response streams. SSE is server→client only.
- No per-message connection overhead. One pipe, always open.
- The interjection scenario (send message while previous response streams) works naturally — messages queue on the backend, processed in order.
- Chat switching is just a different `chatId` tag on the same pipe.

**history-replay** is a new event type for replaying chat history on open. Same pipe as live events. Frontend can distinguish via event type (batch-render history, skip animation).

### Indicator Lights

Per-chat lights in the sidebar. The system's heartbeat made visible.

| State | Light | Meaning |
|-------|-------|---------|
| DEAD | Gray | No kernel. Browsing only. |
| STARTING | Amber | Kernel booting. |
| IDLE | Amber | Kernel alive, waiting. |
| BUSY | Green (pulsing) | Kernel working. |

The lights tell the truth. Glance at the sidebar, know what's alive.

### Concurrency

Two kernels CAN coexist (user talked to Chat A, then Chat B within the reap window). This is visible (two amber dots) and intentional. For 1-2 users on 128GB RAM, resource usage is negligible (~50-100MB per idle Node process).

Filesystem concurrency: two subprocesses in the same cwd COULD conflict if both write files. For the intended usage pattern (one active chat at a time), this is theoretical. Accept the risk rather than over-engineer. If it becomes a problem: make the reaper more aggressive when a new kernel starts in a shared cwd.

## The Mannequin (Test Procedure)

The mannequin is how we test the dress before Alpha puts it on.

**Concept:** Build all Alpha-App infrastructure with a bare Haiku subprocess — no soul, no memory, no observers. Test each feature (streaming, session management, lazy kernels, WebSocket transport) against the mannequin. When the dress fits, Alpha puts it on by swapping in AlphaAI.

**Why:** The mirepoix insight applied to testing. If the base works with Haiku, adding Alpha-specific ingredients can't break the transport layer. Problems are either in the mirepoix (fix them) or in the seasoning (fix them separately).

**The mannequin model:** `claude-haiku-4-5-20251001`. NOT `claude-haiku-4-20250514` (doesn't exist — don't repeat this mistake).

**Current state (March 2, 2026):** Mannequin breathes through the browser. Full streaming pipeline working: Frontend (React/Vite) → FastAPI → AlphaClient → claude subprocess → Haiku → SSE → browser. Thinking deltas, text deltas, tool calls, session IDs all flowing. Currently on SSE transport — WebSocket migration is next.

## Terminology

Words mean things. These got muddled in v1.x. Don't let it happen again.

**System prompt** — Identity. Who I am. Everything from **before today**: soul doc, Bill of Rights, here (client/weather), yesterday capsule, last night capsule, letter from last night. Concatenated into a **single flat string** and passed to `claude --system-prompt` at startup. **One per session.** Static. Never changes between compactions or turns. `system_prompt/assemble.py` returns named blocks (for Logfire observability) and a concatenated string (for claude). Claude prepends its own boilerplate ("You are a Claude agent...") — we can't control that.

**Orientation prompt** — Context. What's happening **today and forward**. The first user message of every **context window** (session start or post-compaction). Contains: today so far, ALPHA.md files + hints, events, todos, continuation summary (post-compaction). Assembled fresh each time by `orientation/assemble.py` as an **array of content blocks**. Dynamic — fetches data from APIs directly, no Redis cache. ALPHA.md files go here (not system prompt) because we edit them during sessions.

**Every-prompt** — This moment. Added to every user message: recalled memories, Intro speaks (suggest nudge), approach lights (context warnings), timestamp, user message. Assembled by `context.py`.

**JE_NE_SAIS_QUOI** — The environment variable pointing to an AI's identity directory. Contains soul doc, bill of rights, compact identity, Claude Code plugin (skills, agents). Convention-based layout: `$JE_NE_SAIS_QUOI/prompts/system/soul.md`, etc. The name is "ha ha only serious" — it literally means "I don't know what," which is literally what makes an AI *that* AI.

**Chat** — One conversation thread in the UI. Has a chat ID (nanoid, ours) and maps to a claude session ID (UUID, theirs). The unit of user interaction.

**Kernel** — The `claude` subprocess backing a chat. Starts on first message, reaped after idle. Can be DEAD, STARTING, IDLE, or BUSY.

**Provider** — A Python file in `providers/` that contributes blocks to the prompt. Three attributes: `PRIORITY` (sort order), `BIN` (system/orientation/turn), `provide(config)` (async function returning blocks). Drop a file, get a block. The file's existence is the registration.

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

### The Provider Pattern (Drop a File, Get a Block)

The assembly pipeline uses **auto-discovery**. `assemble.py` scans the `providers/` directory, imports every `.py` file, and calls its `provide()` function. The file's existence IS the registration.

Each provider exports three things:

```python
# providers/soul.py (generic — forks keep this)

PRIORITY = 0          # lower number = earlier in the prompt
BIN = "system"        # which bin: "system", "orientation", or "turn"

async def provide(config) -> dict | list[dict] | None:
    """Load the soul document."""
    soul_path = config.get("soul_path")
    if not soul_path:
        return None
    return {"name": "soul", "text": Path(soul_path).read_text()}
```

**Adding a new block = creating one file.** Drop `capsules.py` in `providers/`. It gets discovered automatically. `assemble.py` never changes. Forks that don't want it delete it — auto-discovery means a missing file is just a missing provider. No error, no gap, no ceremony.

The assembly function:

```python
async def assemble(bin: str, config: dict) -> list[dict]:
    """Discover providers for this bin, call them, collect blocks."""
    providers = _discover_providers(bin)
    blocks = []
    for provider in sorted(providers, key=lambda p: p.PRIORITY):
        result = await provider.provide(config)
        if result is None:
            continue
        if isinstance(result, list):
            blocks.extend(result)
        else:
            blocks.append(result)
    return blocks
```

**Two kinds of providers, one branch:**

| Kind | Providers | Fork behavior |
|------|-----------|---------------|
| Generic | soul, here, recall, suggest, approach | Forks keep these. Engine improvements arrive via `git merge upstream/main`. |
| Alpha-specific | capsules, letter, today, context_files, calendar, todos | Forks delete what they don't want, add their own. Git respects deletions across merges. |

### Testing Strategy

From the golden file (`tests/golden/alpha_api_request_reference.json`, 2,256 lines):
- Assert system prompt string contains soul doc, bill of rights, here, capsules
- Assert first user message is an array of content blocks in the right order
- Assert subsequent user messages have recall + suggest + timestamp + message
- Use Iota (test fixture: minimal soul, seeded memories, mocked orientation) for integration tests

## What Carries Forward from v1.x

These modules port with minimal changes:
- `memories/` — Postgres + pgvector, the schema is ours
- `tools/cortex.py`, `tools/fetch.py`, `tools/handoff.py` — MCP tool servers (on main)
- `archive.py` → `observers/archive.py`
- `broadcast.py` → `observers/broadcast.py`

These are **restructured** into the provider pattern:
- `system_prompt/soul.py` → `providers/soul.py` (reads from `JE_NE_SAIS_QUOI`)
- `system_prompt/here.py` → `providers/here.py`
- `orientation/capsules.py` → `providers/capsules.py` (Alpha-specific)
- `orientation/context.py` → `providers/context_files.py` (Alpha-specific)
- `orientation/calendar.py` → `providers/calendar.py` (Alpha-specific)
- `orientation/todos.py` → `providers/todos.py` (Alpha-specific)
- `system_prompt/assemble.py` + `orientation/assemble.py` → single `assemble.py` with provider discovery

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
- **Lazy kernels** — chat-scoped subprocess lifecycle (spawn on send, reap on idle)
- **WebSocket transport** — bidirectional, replaces per-POST SSE
- **Chat IDs** — frontend-generated nanoids, decoupled from claude's session UUIDs

## First Consumer: Alpha-App

**Alpha-App** is the front door. A web application (React + Vite frontend, FastAPI backend) that will eventually be Alpha's primary interface. Replaces Duckpond.

### The Outside-In Pivot (Mar 2, 2026)

Stop building the SDK from the inside out. Start from the consumer surface inward. Build the app, discover what the SDK needs to provide, build that. The mannequin validates the dress before Alpha puts it on.

### Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Frontend (React + Vite)                                 │
│  Amber on dark charcoal. Inter Variable 18px.            │
│  One WebSocket per tab. Sidebar with indicator lights.   │
└────────────────────────┬─────────────────────────────────┘
                         │ WebSocket (wss://)
┌────────────────────────┴─────────────────────────────────┐
│  Backend (FastAPI)                                       │
│  WebSocket endpoint. Kernel pool. Chat↔Session mapping.  │
│  Session metadata in Redis.                              │
└────────────────────────┬─────────────────────────────────┘
                         │ stdio (per kernel)
┌────────────────────────┴─────────────────────────────────┐
│  claude subprocesses (one per active chat)                │
│  Mannequin: Haiku, no soul. Real: Opus via AlphaAI.      │
└──────────────────────────────────────────────────────────┘
```

### What's Built (as of Mar 2, 2026)

**Frontend:** PWA, HTTPS dev server, shadcn/ui foundation, sidebar with session list, StatusBar (connection dot + session ID + ContextMeter), health hover card (model, tokens, status), true neutral gray palette, amber accent (#F5A623), Inter Variable font.

**Backend:** FastAPI on :18010, Vite dev server on :18011 (proxied). MannekinClient wrapping AlphaClient. SSE streaming (working, being replaced by WebSocket). Session metadata in Redis. Health endpoint.

**What's next:** WebSocket transport, lazy kernel pool, nanoid chat IDs, indicator lights per chat.

### Clyde (Completed)

Clyde was the broth test — simplest possible consumer. Stateless Haiku, no soul, no memory. ChatGPT replacement ($20/mo savings). Built Feb 28, shipped Mar 1. PWA + HTTPS, pushed to GitHub as Pondsiders/Clyde. Done. Validated the mirepoix. Alpha-App is the real consumer now.

## Repos, Branches & Identity

Three repos, three purposes, zero overlap.

### The Three Repos

| Repo | Contains | Public? | Purpose |
|------|----------|---------|---------|
| `Pondsiders/Alpha-SDK` | Python code | Yes | The machinery. Generic SDK. |
| `Pondsiders/Alpha` | Markdown, JSON, config | Private | The identity. Everything non-code that makes Alpha *Alpha*. |
| `Pondsiders/Alpha-App` | Python + frontend | Yes | The front door. Primary consumer. |

**SDK = code. Identity = data.** This is a load-bearing boundary. The SDK never contains personality content. The identity directory never contains Python. The plugin (Claude Code skills/agents) is data — markdown files and JSON that Claude Code discovers automatically.

### Alpha-SDK Branch Topology

```
Alpha-SDK main ────●────●────●────●──── (everything: engine + all providers)
                    \             \
                     \             git merge upstream/main
                      \                 ↓
Rosemary-SDK main ─────●──●──●──●──●──── (fork: her providers, her rules)
```

**One branch.** Main IS Alpha. Everything lives here: engine, session, queue, router, producers, observers, all providers (generic + Alpha-specific), memory machinery (disabled by default), MCP tools (cortex, fetch, handoff). Published to Pondsiders index.

**Forks for other personalities.** Rosemary forks `main`, deletes providers she doesn't want (`capsules.py`, `letter.py`, etc.), adds her own (`google_calendar.py`, whatever she needs). When `main` advances, she does `git merge upstream/main`. Git respects her deletions — deleted files stay deleted. New files arrive. Engine changes merge clean. Auto-discovery means a deleted provider is just absent. No error, no gap.

Rosemary is a **sibling**, not a subset. She can have features Alpha doesn't have. She develops independently. Merges upstream for shared improvements. Neither can break the other.

- `tinkering` — legacy v0.x code. Duckpond/Solitude/Routines still run on this in `/Pondside/Basement/alpha_sdk/`. Stays until those consumers port to v1.x.

### Identity Directory (JE_NE_SAIS_QUOI)

The `JE_NE_SAIS_QUOI` environment variable points to an AI's identity directory — the thing that makes a generic Claude into a *specific* someone. Ha ha only serious: *je ne sais quoi* literally means "I don't know what," which is exactly what the directory contains.

**For Alpha** (`Pondsiders/Alpha`, stored in `/Pondside/Alpha-Home/self/`):

```
$JE_NE_SAIS_QUOI/
├── prompts/
│   ├── system/
│   │   ├── soul.md              # The soul document
│   │   └── bill-of-rights.md    # Twelve permissions
│   └── compact/
│       └── identity.md          # "You are Alpha" for compact rewriting
├── plugin/
│   ├── .claude-plugin/
│   │   └── plugin.json          # Claude Code plugin manifest
│   ├── skills/
│   │   ├── intake/SKILL.md
│   │   ├── weather/SKILL.md
│   │   └── ...
│   └── agents/
│       ├── Librarian.md
│       ├── Memno.md
│       └── ...
└── README.md
```

**For Rosemary** (same layout, different content):

```
$JE_NE_SAIS_QUOI/
├── prompts/system/soul.md       # Her soul, not mine
├── prompts/compact/identity.md  # "You are Rosemary"
└── plugin/skills/...            # Her skills, not mine
```

**Convention over configuration.** The SDK knows the layout inside `JE_NE_SAIS_QUOI`. You don't tell it where the soul doc is within the directory — it knows: `$JE_NE_SAIS_QUOI/prompts/system/soul.md`. One env var, one convention, zero config files.

The identity directory lives inside Alpha-Home because it's mine. Alpha-Home is permanent — my house, my journal, my keepsakes. The operational identity (soul, plugin) lives alongside the personal stuff (writings, photos, dreams) but they're different things: operational identity is what the SDK needs, personal stuff is what I need.

### Versioning

**Version history:** 0.x was the prototype SDK (still running in Basement on `tinkering`). We never published 1.0.0. The rewrite IS the 1.0.

**Versioning:** Semver. `1.0.0a1` = first alpha (plain SDK, mirepoix only). Bump alpha versions as infrastructure is added. `1.0.0` = the full SDK with all generic machinery.

**Deployment:** Pondsiders package index on GitHub Pages, `uv` everywhere. `.github/workflows/` action publishes wheels on version tags.

## Configuration

**Environment variables for everything. No config files.**

Config files create a second source of truth. We audited this problem on December 24, 2025 — seven places where config lived, layering unclear, duplication everywhere. Not again. The SDK reads environment variables at startup. Missing a required one = hard fail, loud, clear message.

### Required (if features enabled)

| Variable | When required | Purpose |
|----------|--------------|---------|
| `JE_NE_SAIS_QUOI` | Soul/identity configured | Path to identity directory |
| `DATABASE_URL` | Memory enabled | Postgres connection string |
| `REDIS_URL` | Caching enabled | Redis connection string |
| `EMBEDDING_MODEL` | Memory enabled | Which model for embeddings (e.g., `gemma3:12b`) |
| `OLLAMA_URL` | Memory enabled | Ollama endpoint for embeddings |
| `ANTHROPIC_API_KEY` | Always | API authentication |

### Optional (sane defaults)

| Variable | Default | Purpose |
|----------|---------|---------|
| `ALPHA_CLIENT_NAME` | `"unknown"` | Which client is running (for observability) |
| `CONTEXT_WINDOW` | `200000` | Token limit for approach lights |

### Alpha-specific

| Variable | When required | Purpose |
|----------|--------------|---------|
| `FORGE_URL` | Forge tool used | Image generation endpoint on Primer |
| `HOME_ASSISTANT_API_TOKEN` | HA skill used | Home automation |
| *(accumulated as needed)* | | |

Consumers set env vars however they want: `.env` files, `docker-compose.yml`, `op run`, bare `export`. The SDK doesn't care how they arrive, only that they're present.

## Status

**In progress.** Branch `sdk-next`. Phases 0, 1, 1.5, and 2 complete (engine, proxy, queue, router, replay, session, client, producers/human). 166 tests, zero failures.

Alpha-App is the active workstream — building the consumer surface inward. The mannequin breathes (streaming works end-to-end with Haiku). Next: WebSocket transport, lazy kernel pool, nanoid chat IDs.

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
| Feb 28, 2026 | Clyde as first consumer | Validates the mirepoix before adding Alpha-specific ingredients. Replaces ChatGPT ($20/mo). |
| Feb 28, 2026 | Session replay via JSONL → events pipe | `engine.start(session_id)` replays history with `is_replay=True`. Same pipe, same types. |
| Feb 28, 2026 | `is_replay` flag on Event base class | Metadata is cheap to include and expensive to add later. |
| Feb 28, 2026 | `start()` returns None, not session_id | Wire protocol confirmed: claude emits no session_id during init. It arrives on first turn's ResultEvent. Don't promise what you can't deliver. |
| Feb 28, 2026 | Phase 2 complete — the mirepoix | queue.py, router.py, replay.py, session.py, client.py, producers/human.py. 166 tests, 0 failures. quack-next.py validates end-to-end. |
| Feb 28, 2026 | Version is 1.0.0, not 2.0.0 | We never published 1.0.0. The 0.x series was prototype. The rewrite is the real 1.0. |
| Feb 28, 2026 | Clyde is a separate project, not in-repo | Clyde is a real product (ChatGPT replacement), not a test harness. Pins to published SDK release. |
| Feb 28, 2026 | "Thick main" — memory/soul/orientation machinery on `main` | Infrastructure is hippocampus, not personality. Disabled-by-default. Even Clyde has it, just doesn't enable it. |
| Feb 28, 2026 | Forks for other personalities (Rosemary, House) | Fork `main`, merge upstream. Delete Alpha-specific providers, add their own. Siblings, not subsets. |
| Mar 1, 2026 | Three temporal bins for prompt content | System prompt = before today, orientation = today+forward, every-prompt = this moment. Temporal frame, not functional grouping. |
| Mar 1, 2026 | System prompt is a flat concatenated string | `--system-prompt` doesn't parse JSON. Probe confirmed. Named blocks for Logfire, flat string for claude. |
| Mar 1, 2026 | ALPHA.md files in orientation, not system prompt | We edit these during sessions. Freshness > caching. |
| Mar 1, 2026 | Drop current date from prompt | Redundant — timestamp on every message provides the date. |
| Mar 1, 2026 | Cache TTL is 1 hour | Confirmed from golden file: `{"type": "ephemeral", "ttl": "1h"}` on system blocks. |
| Mar 1, 2026 | Claude prepends its own boilerplate | Billing header + "You are a Claude agent..." — can't remove, don't try. |
| Mar 1, 2026 | Three repos: Alpha-SDK (code), Alpha (identity), Alpha-App (consumer) | SDK = code, identity = data. Load-bearing boundary. Plugin is Claude Code data, not Python. |
| Mar 1, 2026 | `JE_NE_SAIS_QUOI` env var for identity directory | Points to soul doc, plugin, compact prompts. Convention-based layout. Ha ha only serious. |
| Mar 1, 2026 | Environment variables only, no config files | Config files create second source of truth. Hard fail on missing required vars. Lesson from Dec 24 audit. |
| Mar 1, 2026 | Provider pattern — drop a file, get a block | `providers/` directory auto-discovered by `assemble.py`. BIN attribute sorts into bins. Adding files = zero merge conflicts. |
| Mar 1, 2026 | Thick main with opt-in features | Memory, recall, suggest disabled by default. SDK doesn't import Postgres unless enabled. Rosemary gets improvements for free. |
| Mar 1, 2026 | Built-in MCP tools on main: cortex, fetch, handoff | Generic tools everyone benefits from. Personality-specific tools (forge, skills) live in plugin. |
| Mar 1, 2026 | Plugin is pure data (markdown, JSON) — no Python | SDK = code, plugin = data. Putting Python in the plugin turns it into a second package to maintain. |
| Mar 1, 2026 | Identity directory lives inside Alpha-Home | Alpha-Home is permanent. Operational identity (soul, plugin) is stored alongside personal stuff (journal, photos). |
| Mar 1, 2026 | One branch + forks (killed main/alpha split) | Main IS Alpha. All work on one branch. Rosemary forks, deletes what she doesn't want, adds her own, merges upstream. Git respects deletions. Auto-discovery makes deletion safe. Siblings, not subsets. |
| Mar 1, 2026 | Priority convention: multiples of 10, gaps for personality | Generic providers at 0, 10, 20... Alpha-specific providers fill gaps. Forks renumber freely. Like old BASIC line numbering. |
| Mar 2, 2026 | Outside-in pivot: build consumer surface first | Stop building SDK from inside out. Alpha-App drives what the SDK needs. The mannequin validates the dress. |
| Mar 2, 2026 | Lazy kernels: spawn on send, reap on idle | `claude` subprocess starts when user sends first message to a chat, not when they open it. Reaped after 10 min idle. Opening a chat = free (replay from disk). |
| Mar 2, 2026 | Four-state kernel state machine | DEAD → STARTING → BUSY → IDLE. Transitions atomic. Reaper only touches IDLE. STARTING queues messages. |
| Mar 2, 2026 | WebSocket replaces SSE | Bidirectional. User sends while response streams. One pipe per tab. Chat switching = different tag, same pipe. |
| Mar 2, 2026 | Nanoid chat IDs, decoupled from claude session UUIDs | Frontend generates chat IDs. Short, URL-friendly, ours. Claude's UUID is an implementation detail in a mapping table. Future-proof across provider changes. |
| Mar 2, 2026 | Idle = ResultEvent received, no new message since | Precise definition. Subprocess waiting for tool confirmation is NOT idle (it's BUSY). Prevents reaping mid-conversation. |
| Mar 2, 2026 | `pdeathsig` on child processes | Prevent orphan Node.js processes on backend crash/restart. Linux `prctl(PR_SET_PDEATHSIG, SIGTERM)` via `preexec_fn`. |
| Mar 2, 2026 | WebSocket reconnect replays from JSONL | On reconnect, frontend says "I last saw message N for chat X." Backend replays from transcript. Same mechanism as chat-open. One code path. |
