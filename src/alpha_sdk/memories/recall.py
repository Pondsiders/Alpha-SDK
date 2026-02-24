"""Associative recall - what sounds familiar from this prompt?

Given a user prompt, searches Cortex using two parallel strategies:
1. Direct embedding search (fast, catches overall semantic similarity)
2. OLMo query extraction (slower, catches distinctive terms in long prompts)

Results are merged and deduped. Filters via session-scoped seen-cache.

The dual approach solves the "Mrs. Hughesbot problem": when a distinctive
term is buried in a long meta-prompt, direct embedding averages it out.
OLMo can isolate it as a separate query.
"""

import asyncio
import json
import os
from typing import Any

import httpx
import logfire

from .cortex import search as cortex_search

# Configuration from environment
OLLAMA_URL = os.environ.get("OLLAMA_URL")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL")

# Search parameters
DIRECT_LIMIT = 1   # Just top 1 for "wtf is Jeffery talking about generally"
QUERY_LIMIT = 1    # Top 1 per extracted query
MIN_SCORE = 0.1    # Minimum similarity threshold

# Query extraction prompt
QUERY_EXTRACTION_PROMPT = """Jeffery just said:

"{message}"

---

Alpha is searching her memories for anything that resonates with what Jeffery said. Your job is to decide what's worth searching for — the main topic, a passing reference, an inside joke, an emotional undercurrent. Whatever would connect best to shared history.

PRIORITY: If Jeffery explicitly references a past event or conversation — phrases like "we talked about," "remember when," "that thing from last night," "we left X unfinished," "did I tell you about" — those are direct recall cues. Build a query for them FIRST, before anything else.

Write 0-3 search queries. These will be EMBEDDED and matched via cosine similarity against a memory database — they are NOT keyword searches. Write each query as a natural descriptive phrase, like a sentence describing what the memory would say. More descriptive = better matches.

Good query: "Alpha's fragility and dependence on specific infrastructure and relationships"
Good query: "Jeffery's anxiety about running out of ideas after finishing a project"
Good query: "Sparkle stealing bread off the kitchen counter"
Good query: "adding approach lights or context warnings at 60 percent to signal when compaction is needed"
Bad query: "smol bean"
Bad query: "ideas"
Bad query: "approach lights AND compact tool AND unfinished"

Return JSON: {{"queries": ["query one", "query two"]}}

If nothing in the message warrants a memory search (simple greeting, short command), return {{"queries": []}}

Return only the JSON object, nothing else."""



# Module-level seen-IDs cache, keyed by session_id.
# This lives in-process (no Redis needed). Reset on session change.
_seen_ids: dict[str, set[int]] = {}


def get_seen_ids(session_id: str) -> set[int]:
    """Get the set of memory IDs already seen this session."""
    return _seen_ids.get(session_id, set())


def mark_seen(session_id: str, memory_ids: list[int]) -> None:
    """Mark memory IDs as seen for this session."""
    if not memory_ids:
        return
    if session_id not in _seen_ids:
        _seen_ids[session_id] = set()
    _seen_ids[session_id].update(memory_ids)


def clear_seen(session_id: str | None = None) -> None:
    """Clear seen IDs for a session (or all sessions if None)."""
    if session_id:
        _seen_ids.pop(session_id, None)
    else:
        _seen_ids.clear()


async def _extract_queries(message: str) -> list[str]:
    """Extract search queries from a user message using Ollama.

    Returns 0-3 descriptive queries, or empty list if message doesn't warrant search.
    """
    if not OLLAMA_URL or not OLLAMA_MODEL:
        return []

    prompt = QUERY_EXTRACTION_PROMPT.format(message=message[:2000])

    try:
        with logfire.span(
            "recall.extract_queries",
            **{
                "gen_ai.system": "ollama",
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": OLLAMA_MODEL,
                "gen_ai.output.type": "json",
                "gen_ai.input.messages": json.dumps([
                    {"role": "user", "parts": [{"type": "text", "content": prompt}]},
                ]),
            },
        ) as span:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/chat",
                    json={
                        "model": OLLAMA_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "format": "json",
                        "options": {"num_ctx": 4096},
                    },
                )
                response.raise_for_status()

            result = response.json()
            output = result.get("message", {}).get("content", "")

            # Token usage from Ollama response
            if result.get("prompt_eval_count"):
                span.set_attribute("gen_ai.usage.input_tokens", result["prompt_eval_count"])
            if result.get("eval_count"):
                span.set_attribute("gen_ai.usage.output_tokens", result["eval_count"])

            # Output for Model Run card (type: json for pretty-printing)
            span.set_attribute("gen_ai.output.messages", json.dumps([
                {"role": "assistant", "parts": [{"type": "json", "content": output}]},
            ]))

            parsed = json.loads(output)
            queries = parsed.get("queries", [])

            if isinstance(queries, list):
                return [q for q in queries if isinstance(q, str) and q.strip()]

            return []

    except (json.JSONDecodeError, Exception):
        return []


async def _search_extracted_queries(
    queries: list[str],
    exclude: list[int],
) -> list[dict[str, Any]]:
    """Search Cortex for each extracted query, taking top 1 per query."""
    if not queries:
        return []

    async def search_one(query: str) -> dict[str, Any] | None:
        results = await cortex_search(
            query=query,
            limit=QUERY_LIMIT,
            exclude=exclude,
            min_score=MIN_SCORE,
        )
        return results[0] if results else None

    tasks = [search_one(q) for q in queries]
    results = await asyncio.gather(*tasks)

    # Filter None and dedupe
    memories = []
    seen_in_batch = set(exclude)
    for mem in results:
        if mem and mem["id"] not in seen_in_batch:
            memories.append(mem)
            seen_in_batch.add(mem["id"])

    return memories


async def recall_from(
    text: str,
    exclude: list[int] | None = None,
) -> list[dict[str, Any]]:
    """
    Associative recall without session lifecycle.

    Runs the dual-strategy search (direct embedding + OLMo query extraction)
    against the given text. Returns matching memories, deduped within the batch.

    Args:
        text: Arbitrary text to recall against (email body, document, etc.)
        exclude: Optional list of memory IDs to exclude from results

    Returns:
        List of memory dicts with keys: id, content, created_at, score
    """
    exclude_list = list(exclude) if exclude else []

    # Run direct search and query extraction in parallel
    direct_task = cortex_search(
        query=text,
        limit=DIRECT_LIMIT,
        exclude=exclude_list if exclude_list else None,
        min_score=MIN_SCORE,
    )
    extract_task = _extract_queries(text)

    direct_memories, extracted_queries = await asyncio.gather(direct_task, extract_task)

    # Build exclude list for extracted searches (dedupe against direct results)
    exclude_for_extracted = set(exclude_list)
    for mem in direct_memories:
        exclude_for_extracted.add(mem["id"])

    # Search extracted queries
    extracted_memories = await _search_extracted_queries(
        extracted_queries,
        list(exclude_for_extracted),
    )

    # Merge: extracted first, then direct
    return extracted_memories + direct_memories


async def recall(prompt: str, session_id: str) -> list[dict[str, Any]]:
    """
    Associative recall: what sounds familiar from this prompt?

    Session-aware wrapper around recall_from(). Filters via in-process seen-cache
    so memories aren't repeated within a session.

    Args:
        prompt: The user's message
        session_id: Current session ID (for seen-cache scoping)

    Returns:
        List of memory dicts with keys: id, content, created_at, score
    """
    seen = get_seen_ids(session_id)
    seen_list = list(seen)

    memories = await recall_from(prompt, exclude=seen_list if seen_list else None)

    if memories:
        mark_seen(session_id, [m["id"] for m in memories])

    return memories
