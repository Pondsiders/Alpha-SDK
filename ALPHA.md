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

### Core Principle: A Library, Not a Framework

The SDK is a collection of **functions you import and call**. No inversion of control. No event bus. No pipeline. No middleware. The consumer decides what to use, when to call it, and how to compose the results.

```python
from alpha_sdk.claude import Claude
from alpha_sdk.system_prompt import assemble_system_prompt
from alpha_sdk.orientation import assemble_orientation
from alpha_sdk.recall import recall
from alpha_sdk.suggest import suggest
from alpha_sdk.archive import archive_turn
```

The consumer orchestrates. The SDK provides the pieces.

### Claude

The `Claude` class wraps the `claude` subprocess. It's the only stateful object in the SDK.

```python
claude = Claude(
    system_prompt=system_prompt_string,
    session_id=None,           # None = new, string = --resume
    mcp_config=mcp_config_path,
    permission_mode="bypassPermissions",
)
await claude.start()
# Model comes from alpha_sdk.constants.MODEL — not a constructor parameter.
```

Claude owns:

- **The subprocess** — `claude` binary, stdin/stdout/stderr pipes
- **The HTTP proxy** — localhost reverse proxy to Anthropic (compact rewriting, token counting, usage header sniffing). One proxy per subprocess, lifecycle 1:1. Engine-private.
- **The inbox** — `asyncio.Queue`. Anyone can `send()`. A background writer drains to stdin.
- **The fan-out** — Primary consumer iterates `events()`. Eavesdroppers get fire-and-forget notification via `subscribe()`.
- **MCP dispatch** — Routes `control_request` messages to in-process FastMCP tool handlers. Dict in, dict out. The quack-mcp.py pattern.
- **The handoff procedure** — On handoff tool call: wait for turn end → emit `compact_start` → `/compact` → wait for summary → assemble orientation → reorient → emit `compact_end`.

Two independent loops make the duplex real:

```
Writer loop:  inbox.get() → subprocess stdin     (runs continuously)
Reader loop:  subprocess stdout → yield + notify  (runs continuously)
```

Send and receive are independent streams. You can type while the response is still streaming. This is what makes it a phone call, not a series of letters.

#### API

```python
# Input — anyone can send, from anywhere
await claude.send(content_blocks)
await claude.send(content_blocks, source="email")  # source is metadata for logging

# Output — primary consumer
async for event in claude.events():
    ...

# Output — eavesdroppers (fire and forget, don't block the primary consumer)
claude.subscribe(my_callback)    # async def my_callback(event: Event) -> None

# Lifecycle
await claude.start()
await claude.stop()

# State
claude.state          # EngineState enum
claude.session_id     # str | None — discovered from first ResultEvent
```

#### Events

Events are typed dataclasses, same as today:

| Event | When |
|-------|------|
| `InitEvent` | Subprocess ready, capabilities received |
| `UserEvent` | User message echoed back (replay) |
| `AssistantEvent` | Assistant content (text, thinking, tool calls) |
| `StreamEvent` | Streaming delta (text-delta, thinking-delta) |
| `ResultEvent` | Turn complete. Carries session_id, turn count. |
| `SystemEvent` | System messages (compaction, etc.) |
| `ErrorEvent` | Something went wrong |

All events have `is_replay: bool` for distinguishing live vs. replayed history.

### Functions

Everything else is a function. Import what you need. Ignore what you don't.

#### system_prompt.py

Assembles the system prompt — a flat string for `claude --system-prompt`.

```python
from alpha_sdk.system_prompt import assemble_system_prompt

prompt = await assemble_system_prompt()
# Returns: soul doc + bill of rights + here + capsules + letter
# One string. Concatenated. Done.
```

Internally calls `_load_soul()`, `_load_bill_of_rights()`, `_load_here()`, `_load_capsules()`, `_load_letter()`. Each is a function, not a provider class. When we need a new component, we add a function and add a call. `assemble_system_prompt()` is the only public surface.

#### orientation.py

Assembles orientation blocks — the first-turn context for each context window.

```python
from alpha_sdk.orientation import assemble_orientation

blocks = await assemble_orientation()
# Returns: list of content blocks
# today-so-far, ALPHA.md files, events, todos, continuation summary
```

#### recall.py

Associative memory recall — the full pipeline from user input to relevant memories.

```python
from alpha_sdk.recall import recall

memories = await recall(
    user_content,           # text content blocks (+ images)
    session_id=session_id,  # for dedup tracking across the context window
)
# Returns: list of Memory objects, deduplicated, ready for content blocks
```

Recall is not search. Search is a low-level database operation in `memories/` (query in, scored results out). Recall is the associative layer above it:

1. **Query extraction** — Feeds the user's message to the helper LM (currently Gemma 3 12B via Ollama, multimodal) that generates search queries: "what sounds familiar here?" The LM understands conversational context and produces queries suitable for semantic search.
2. **Image handling** — If the message includes images, feeds them to the multimodal helper LM for text descriptions. Those descriptions become additional search queries. So when Jeffery sends a photo, I get memories of similar things I've seen or discussed.
3. **Search** — Sends each generated query to `memories.search()` (hybrid: semantic via pgvector + full-text via tsvector, linearly combined scoring).
4. **Deduplication** — Tracks already-seen memories per context window. If the same topic comes up three times, recall returns three *different* memories each time, not the same one repeated.
5. **Formatting** — Returns memories as structured objects ready to be formatted as content blocks and prepended to the user message.

The consumer calls `recall()` and gets back memories. The consumer decides when to call it (every turn? important turns only?) and how to format the results. The helper LM, dedup tracking, and image processing are recall's internal business.

#### suggest.py

Analyzes a conversation turn and extracts memorables.

```python
from alpha_sdk.suggest import suggest

memorables = await suggest(assistant_content, user_content)
# Returns: list of strings worth remembering
```

Calls the helper LM (currently Gemma 3 12B via Ollama) to identify what's worth storing. The consumer decides when to call it (after each turn? after important turns? never?). The helper LM for suggest and recall is the same — a self-contained chunk that can be replaced wholesale when we're ready to try alternatives (e.g., Claude Holster pattern).

#### archive.py

Writes conversation turns to Postgres for the Scribe.

```python
from alpha_sdk.archive import archive_turn

await archive_turn(session_id, user_content, assistant_content)
# user_content and assistant_content are lists of content blocks
```

#### memories/

The Cortex memory system. The low-level primitives that `recall.py` and the MCP tools build on.

- `memories/db.py` — Direct Postgres operations. `search()` is the hybrid query engine (semantic via pgvector + full-text via tsvector, linear combination scoring). This is the function `recall` calls for each generated query.
- `memories/cortex.py` — High-level memory API: store, search, recent. Used by MCP tools and directly by consumers.
- `memories/embeddings.py` — Embedding generation (model configured via `EMBEDDING_MODEL` + `OLLAMA_URL`).

#### tools/

FastMCP tool definitions. Claude dispatches to these internally — no external MCP server processes. Consumers pass them at construction; Claude handles all MCP traffic via the `type: "sdk"` control_request protocol (proven in quack-mcp.py).

- `tools/cortex.py` — Memory tools (store, search, recent)
- `tools/fetch.py` — Web fetching
- `tools/handoff.py` — Context window management. Handled entirely inside Claude: on handoff call, Claude waits for the current turn to finish, emits `compact_start`, sends `/compact`, waits for the summary, assembles fresh orientation, sends the reorientation prompt, emits `compact_end`. Consumers get compact for free.

### How Consumers Compose

The SDK doesn't know how it'll be used. The consumer composes.

The model is not a consumer choice. It's an SDK constant — set once in the SDK (e.g. `alpha_sdk.constants`), used everywhere. Upgrading the model means releasing a new SDK version. All consumers get it.

**Alpha-App** (full experience):

```python
# At startup
system_prompt = await assemble_system_prompt()
claude = Claude(system_prompt=system_prompt, mcp_config=MCP_CONFIG)
claude.subscribe(archive_eavesdropper)
await claude.start()

# On user message (in the WebSocket handler)
memories = await recall(user_text)
blocks = recall_blocks(memories) + intro_blocks + [user_block]
await claude.send(blocks)

# Stream response to browser
async for event in claude.events():
    ws.send(format_event(event))

# After turn
memorables = await suggest(assistant_text, user_text)
```

**Solitude** (autonomous nighttime breathing):

```python
system_prompt = await assemble_system_prompt()
claude = Claude(system_prompt=system_prompt, mcp_config=MCP_CONFIG)
claude.subscribe(archive_eavesdropper)
await claude.start()

# Each breath — the scheduler sends a canned prompt
await claude.send([breath_block])

# Read the response (even though nobody's watching live)
async for event in claude.events():
    log(event)  # or just drain it

# After turn
memorables = await suggest(assistant_text, breath_prompt)
```

**Capsules** (constrained one-shot):

```python
system_prompt = await assemble_system_prompt()  # or a capsule-specific prompt
claude = Claude(system_prompt=system_prompt)
await claude.start()

# One message, one response, goodbye
await claude.send([capsule_prompt_block])
async for event in claude.events():
    collect(event)

await claude.stop()
```

No recall. No suggest. No eavesdroppers. Just send and receive.

### The Package

```
src/alpha_sdk/
├── __init__.py
├── claude.py              # The subprocess. Inbox, fan-out, proxy, events.
├── system_prompt.py       # Assembles system prompt string.
├── orientation.py         # Assembles orientation content blocks.
├── recall.py              # Memory recall (hybrid search).
├── suggest.py             # Memorables extraction.
├── archive.py             # Turn archiving to Postgres.
│
├── memories/              # Cortex internals (ported from v1.x)
│   ├── db.py
│   ├── cortex.py
│   ├── embeddings.py
│   └── ...
│
└── tools/                 # MCP tool servers (ported from v1.x)
    ├── cortex.py
    ├── fetch.py
    └── handoff.py
```

Six core files. A memories package. A tools package. That's it.

## Protocol

Based on quack-wire.py observations (Feb 26):

- **Spawn:** `claude --output-format stream-json --input-format stream-json --model MODEL [--mcp-config path]`
- **Init:** claude emits a capabilities advertisement (tools, servers, model, commands)
- **Input:** JSON messages on stdin (user messages, init handshake, control responses)
- **Output:** Newline-delimited JSON events on stdout (assistant messages, results, system events, control requests)
- **Built-in tools:** Execute inside claude. We see results, not execution. (Bash, Read, Edit, etc.)
- **SDK tools:** claude sends `control_request` with subtype `mcp_message`. We dispatch to in-process FastMCP handlers and return `control_response` with the result. Claude thinks it's talking to a real MCP server — it is, it just lives in our process.

### The Four I/O Channels

The claude subprocess has four communication channels:

1. **stdin** — JSON messages in (user messages, init handshake, control responses including MCP results)
2. **stdout** — JSON events out (assistant messages, results, system events, control requests including MCP calls)
3. **stderr** — Diagnostic output (drained in background, not parsed)
4. **HTTP to Anthropic** — API requests via `ANTHROPIC_BASE_URL` (inference, compaction)

Claude manages all four. Channels 1-3 are direct subprocess pipes. Channel 4 is intercepted by a localhost HTTP proxy that Claude starts before spawning the subprocess. The proxy:

- **Rewrites compact requests** — Replaces claude's default summarizer identity, compact instructions, and continuation instruction with the identity's versions. Three-phase surgical rewrite.
- **Counts tokens** — Fire-and-forget echo to `/v1/messages/count_tokens` on every API request. Tracks high-water mark.
- **Sniffs usage headers** — Extracts `anthropic-ratelimit-unified-{7d,5h}-utilization` from Anthropic's response headers.

The proxy is private to Claude — no other module imports or knows about it.

## Prompt Architecture

Three temporal bins. The concept is right even though the implementation is simpler than before.

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

### Implementation

**System prompt:** `system_prompt.py` → `assemble_system_prompt()`. Returns a flat string. Claude passes it to `--system-prompt` at startup.

**Orientation:** `orientation.py` → `assemble_orientation()`. Returns content blocks. The consumer sends these as (or prepended to) the first user message of each context window.

**Every-prompt:** The consumer assembles this. Calls `recall()`, formats the results, adds timestamp, adds user message. The SDK provides the pieces; the consumer does the composition.

### Why This Sort

**System prompt = frozen history.** Yesterday is yesterday. The soul doesn't change mid-session. These earn the identity-weight training signal and are always cached.

**Orientation = live context.** Today changes. ALPHA.md gets edited. Todos get completed. Fetched fresh at each context window boundary.

**Every-prompt = ephemeral.** Memories are query-specific. Timestamps are per-message.

## Terminology

Words mean things.

**System prompt** — Identity. Who I am. Everything from before today. One per session, flat string, passed to `--system-prompt`.

**Orientation** — Context. What's happening today and forward. First user message of each context window. Array of content blocks. Fetched fresh.

**Claude** — The Python class wrapping the `claude` subprocess. Owns the process, the proxy, the inbox, and the fan-out.

**Chat** — Alpha-App concept. Lifecycle, state machine, reaper. Owns a Claude. Defined in KERNEL.md.

**Kernel** — Synonym for "the Claude subprocess backing a chat." Same thing, different angle.

**JE_NE_SAIS_QUOI** — Env var pointing to an AI's identity directory. Soul doc, bill of rights, compact identity, plugin.

**Context window** — One continuous conversation before compaction. A session may span multiple.

**Turn** — One user message + one assistant response.

**HUD** — Dead. Don't resurrect.

## First Consumer: Alpha-App

Alpha-App is the front door. React + Vite frontend, FastAPI backend. Architecture lives in KERNEL.md.

The key relationship: **Chat wraps Claude.** Chat handles lifecycle (state machine, reaper, Holster). Claude handles the subprocess and I/O. The Alpha-App backend calls SDK functions (recall, suggest, orientation) in its WebSocket handler. The SDK doesn't know Alpha-App exists.

See KERNEL.md for: Chat class, Holster pattern, state machine, WebSocket protocol, nanoid IDs, indicator lights, serialization.

## Repos, Branches & Identity

Three repos, three purposes, zero overlap.

| Repo | Contains | Public? | Purpose |
|------|----------|---------|---------|
| `Pondsiders/Alpha-SDK` | Python code | Yes | The machinery. Generic SDK. |
| `Pondsiders/Alpha` | Markdown, JSON, config | Private | The identity. Everything non-code that makes Alpha *Alpha*. |
| `Pondsiders/Alpha-App` | Python + frontend | Yes | The front door. Primary consumer. |

**SDK = code. Identity = data.** Load-bearing boundary.

### Branch Topology

One branch. Main IS Alpha. Forks for siblings (Rosemary, House). Git respects deletions across merges. Each sibling develops independently, merges upstream for shared improvements.

### Identity Directory (JE_NE_SAIS_QUOI)

```
$JE_NE_SAIS_QUOI/
├── prompts/
│   ├── system/
│   │   ├── soul.md
│   │   └── bill-of-rights.md
│   └── compact/
│       └── identity.md
├── plugin/
│   ├── skills/
│   └── agents/
└── README.md
```

Convention over configuration. The SDK knows the layout. One env var, one convention, zero config files.

## Configuration

Environment variables for everything. No config files.

### Required (if features used)

| Variable | When | Purpose |
|----------|------|---------|
| `JE_NE_SAIS_QUOI` | Soul/identity | Path to identity directory |
| `DATABASE_URL` | Memory/archive | Postgres connection string |
| `EMBEDDING_MODEL` | Memory | Which model for embeddings |
| `OLLAMA_URL` | Memory | Ollama endpoint |
| `ANTHROPIC_API_KEY` | Always | API authentication |

### Optional

| Variable | Default | Purpose |
|----------|---------|---------|
| `ALPHA_CLIENT_NAME` | `"unknown"` | Which client (for observability) |
| `CONTEXT_WINDOW` | `200000` | Token limit for approach lights |

## Status

**Architecture settled. Ready to build.** The Claude class absorbs engine and proxy. Queue, router, session, and client are dead (killed Mar 4). Tools dispatch in-process via FastMCP. Handoff lives inside Claude.

Alpha-App is the first consumer. KERNEL.md describes the consumer architecture.

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| Feb 26 | Prototype raw `claude` stdio | quack-raw.py, quack-wire.py, quack-mcp.py |
| Feb 27 | ~~Primary + sidecar engine pool~~ | ~~Different models for different workloads.~~ Dead — recall/suggest use Gemma, not a claude sidecar. |
| Feb 27 | System prompt vs orientation terminology | System = identity (per session). Orientation = context (per window). |
| Feb 27 | HTTP proxy is Engine-private | Fourth I/O channel. Born/dies with subprocess. |
| Feb 28 | Mirepoix reframe | Generic foundation, personalization above. |
| Feb 28 | Version is 1.0.0 | We never published 1.0.0. The rewrite IS the 1.0. |
| Mar 1 | Three temporal bins | System = before today, orientation = today+, every-prompt = this moment. |
| Mar 1 | System prompt is flat string | `--system-prompt` doesn't parse JSON. Probe confirmed. |
| Mar 1 | Three repos | SDK = code, identity = data, app = consumer. |
| Mar 1 | `JE_NE_SAIS_QUOI` | Identity directory env var. Ha ha only serious. |
| Mar 1 | Env vars only, no config files | Lesson from Dec 24 audit. |
| Mar 2 | Outside-in pivot | Build consumer surface first. Mannequin validates dress. |
| Mar 2 | Lazy kernels | Spawn on send, reap on idle. See KERNEL.md. |
| Mar 2 | WebSocket replaces SSE | Bidirectional. One pipe per tab. |
| Mar 2 | Nanoid chat IDs | Frontend-generated, decoupled from Claude UUIDs. |
| Mar 4 | **Library, not framework** | Functions you import and call. No pipeline, no event bus, no middleware. The consumer composes. |
| Mar 4 | **Claude owns inbox + fan-out** | Input queue and output eavesdroppers live on Claude, not a separate Session. |
| Mar 4 | **Proxy absorbed into Claude** | One proxy per subprocess, lifecycle 1:1. No shared proxy. |
| Mar 4 | **Kill producers/observers/providers** | Beautiful engineering for a problem we don't have. Just functions. |
| Mar 4 | **Kill Session, Router, Queue** | Chat IS the session (Alpha-App). Solitude composes differently. No universal abstraction needed. |
| Mar 4 | **Kill auto-discovery** | `system_prompt.py` just assembles it. Add a function, add a call. Refactor when it hurts. |
| Mar 5 | Recall is not search | Recall = full associative pipeline (query extraction → image handling → search → dedup → format). Search = low-level database op. |
| Mar 5 | Helper LM is Ollama/Gemma, not pluggable | Self-contained chunk. Replace wholesale when ready. Don't abstract. |
| Mar 5 | Model is SDK constant | Not a constructor parameter. One place to change, SDK release to upgrade. |
| Mar 5 | **MCP dispatch inside Claude** | In-process FastMCP handlers. quack-mcp.py pattern. No external MCP server processes. |
| Mar 5 | **Handoff inside Claude** | Claude owns the full procedure: wait → compact_start → /compact → reorient → compact_end. Consumers get compact for free. |
| Mar 5 | Solitude doesn't use recall | No user prompts, nothing to associate against. |
| Mar 5 | archive_turn takes content block lists | Interface is lists. Storage format is archive.py's business. |
| Mar 5 | Embedding stays on Ollama GPU | Benchmark: 3x serial 33ms (GPU) vs 70ms (CPU). Batch API does 3 in 14ms. Not the bottleneck. |
