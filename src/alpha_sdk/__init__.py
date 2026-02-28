"""Alpha SDK — everything that turns Claude into Alpha."""

__version__ = "2.0.0a0"

from .client import AlphaClient
from .engine import (
    AssistantEvent,
    Engine,
    EngineState,
    ErrorEvent,
    Event,
    InitEvent,
    ResultEvent,
    SystemEvent,
)
from .queue import Message, MessageQueue
from .replay import find_session_path, replay_session
from .router import Observer, Router
from .session import Session

__all__ = [
    # Client
    "AlphaClient",
    # Session
    "Session",
    # Engine
    "Engine",
    "EngineState",
    # Events
    "Event",
    "InitEvent",
    "AssistantEvent",
    "ResultEvent",
    "SystemEvent",
    "ErrorEvent",
    # Queue
    "Message",
    "MessageQueue",
    # Router
    "Router",
    "Observer",
    # Replay
    "replay_session",
    "find_session_path",
]
