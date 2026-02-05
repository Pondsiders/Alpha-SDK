"""alpha_sdk - Everything that turns Claude into Alpha.

Architecture:
- System prompt (soul) passed directly to Claude Agent SDK
- Orientation (context) injected in first user message
- Minimal proxy intercepts only compact prompts for rewriting
"""

from .client import AlphaClient, PermissionMode
from .compact_proxy import CompactProxy
from .observability import configure as configure_observability
from .sessions import SessionInfo, list_sessions, get_session_path, get_sessions_dir

__all__ = [
    # Main client
    "AlphaClient",
    "PermissionMode",
    # Compact proxy (for direct use if needed)
    "CompactProxy",
    # Session discovery
    "SessionInfo",
    "list_sessions",
    "get_session_path",
    "get_sessions_dir",
    # Observability
    "configure_observability",
]
__version__ = "0.3.0"  # Compact proxy for prompt rewriting
