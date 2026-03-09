"""Memory operations — Cortex integration.

Direct Postgres access via asyncpg, embeddings via Ollama.
store/search/recent/get/forget are the foundational operations.
recall and suggest are higher-level functions (ported separately).
"""

from .cortex import store, search, recent, get, forget, health, close
from .db import init_schema
from .embeddings import EmbeddingError

__all__ = [
    "store",
    "search",
    "recent",
    "get",
    "forget",
    "health",
    "close",
    "init_schema",
    "EmbeddingError",
]
