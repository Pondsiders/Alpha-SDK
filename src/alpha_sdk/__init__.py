"""alpha_sdk - Everything that turns Claude into Alpha.

Proxyless architecture:
- System prompt (soul) passed directly to Claude Agent SDK
- Orientation (context) injected in first user message
- No proxy, no canary, no interception
"""

from .client import AlphaClient, PermissionMode
from .observability import configure as configure_observability
from .sessions import SessionInfo, list_sessions, get_session_path, get_sessions_dir

__all__ = [
    # Main client
    "AlphaClient",
    "PermissionMode",
    # Session discovery
    "SessionInfo",
    "list_sessions",
    "get_session_path",
    "get_sessions_dir",
    # Observability
    "configure_observability",
]
__version__ = "0.2.0"  # Proxyless architecture
