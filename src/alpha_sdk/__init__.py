"""Alpha SDK — everything that turns Claude into someone."""

__version__ = "1.0.0"

from .claude import (
    AssistantEvent,
    Claude,
    ClaudeState,
    ErrorEvent,
    Event,
    InitEvent,
    ResultEvent,
    StreamEvent,
    SystemEvent,
    UserEvent,
)
from .system_prompt import assemble_system_prompt, read_soul

__all__ = [
    # The one class
    "Claude",
    "ClaudeState",
    # Events
    "Event",
    "InitEvent",
    "UserEvent",
    "AssistantEvent",
    "ResultEvent",
    "SystemEvent",
    "ErrorEvent",
    "StreamEvent",
    # System prompt
    "assemble_system_prompt",
    "read_soul",
]
