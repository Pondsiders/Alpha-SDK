"""System prompt assembly - weaving threads into Alpha's context."""

from .assemble import assemble
from .soul import get_soul, get_compact, init as init_soul
from .capsules import get_capsules
from .here import get_here, get_hostname
from .context import load_context
from .calendar import get_events
from .todos import get_todos

__all__ = [
    "assemble",
    "get_soul",
    "get_compact",
    "init_soul",
    "get_capsules",
    "get_here",
    "get_hostname",
    "load_context",
    "get_events",
    "get_todos",
]
