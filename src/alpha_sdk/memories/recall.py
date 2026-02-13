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
        logfire.debug("OLLAMA not configured, skipping query extraction")
        return []

    prompt = QUERY_EXTRACTION_PROMPT.format(message=message[:2000])

    with logfire.span(
        "recall.extract_queries",
        **{
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "ollama",
            "gen_ai.request.model": OLLAMA_MODEL,
        }
    ) as span:
        try:
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

            span.set_attribute("gen_ai.usage.input_tokens", result.get("prompt_eval_count", 0))
            span.set_attribute("gen_ai.usage.output_tokens", result.get("eval_count", 0))
            span.set_attribute("gen_ai.response.model", OLLAMA_MODEL)

            parsed = json.loads(output)
            queries = parsed.get("queries", [])

            if isinstance(queries, list):
                valid = [q for q in queries if isinstance(q, str) and q.strip()]
                logfire.debug("Extracted queries", count=len(valid), queries=valid)
                return valid

            return []

        except json.JSONDecodeError as e:
            logfire.warning("Failed to parse OLMo response as JSON", error=str(e))
            return []
        except Exception as e:
            logfire.error("Query extraction failed", error=str(e))
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

    with logfire.span("recall.search_extracted", query_count=len(queries)) as span:
        tasks = [search_one(q) for q in queries]
        results = await asyncio.gather(*tasks)

        # Instrumentation
        query_results = {
            q: (r["id"] if r else None)
            for q, r in zip(queries, results)
        }
        span.set_attribute("query_results", str(query_results))

        # Filter None and dedupe
        memories = []
        seen_in_batch = set(exclude)
        for i, mem in enumerate(results):
            if mem and mem["id"] not in seen_in_batch:
                memories.append(mem)
                seen_in_batch.add(mem["id"])
                logfire.debug(f"Query '{queries[i]}' -> memory #{mem['id']}")
            elif mem:
                logfire.debug(f"Query '{queries[i]}' -> memory #{mem['id']} (deduped)")
            else:
                logfire.debug(f"Query '{queries[i]}' -> no result above threshold")

        return memories


async def recall(prompt: str, session_id: str) -> list[dict[str, Any]]:
    """
    Associative recall: what sounds familiar from this prompt?

    Uses two parallel strategies:
    1. Direct embedding search (fast, semantic similarity)
    2. OLMo query extraction + search (slower, catches distinctive terms)

    Results are merged and deduped. Filters via in-process seen-cache.

    Args:
        prompt: The user's message
        session_id: Current session ID (for seen-cache scoping)

    Returns:
        List of memory dicts with keys: id, content, created_at, score
    """
    with logfire.span("recall", session_id=session_id[:8] if session_id else "none") as span:
        seen = get_seen_ids(session_id)
        seen_list = list(seen)
        logfire.debug("Seen IDs loaded", count=len(seen_list))

        # Run direct search and query extraction in parallel
        direct_task = cortex_search(
            query=prompt,
            limit=DIRECT_LIMIT,
            exclude=seen_list if seen_list else None,
            min_score=MIN_SCORE,
        )
        extract_task = _extract_queries(prompt)

        direct_memories, extracted_queries = await asyncio.gather(direct_task, extract_task)

        span.set_attribute("extracted_queries", extracted_queries)
        span.set_attribute("direct_memory_ids", [m["id"] for m in direct_memories])

        # Build exclude list for extracted searches
        exclude_for_extracted = set(seen_list)
        for mem in direct_memories:
            exclude_for_extracted.add(mem["id"])

        # Search extracted queries
        extracted_memories = await _search_extracted_queries(
            extracted_queries,
            list(exclude_for_extracted),
        )

        span.set_attribute("extracted_memory_ids", [m["id"] for m in extracted_memories])

        # Merge: extracted first, then direct
        all_memories = extracted_memories + direct_memories
        span.set_attribute("total_memories", len(all_memories))

        if not all_memories:
            logfire.info("No memories above threshold")
            return []

        # Mark as seen (in-process, no Redis)
        new_ids = [m["id"] for m in all_memories]
        mark_seen(session_id, new_ids)

        logfire.debug(
            "Recall complete",
            extracted=len(extracted_memories),
            direct=len(direct_memories),
            total=len(all_memories),
        )

        return all_memories
