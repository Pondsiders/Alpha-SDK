"""alpha_sdk - Everything that turns Claude into Alpha."""

from .client import AlphaClient, PermissionMode
from .weave import weave
from .proxy import AlphaProxy
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
    # Lower-level components
    "AlphaProxy",
    "weave",
    "configure_observability",
]
__version__ = "0.1.0"
