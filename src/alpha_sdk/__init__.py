"""Alpha SDK — everything that turns Claude into Alpha."""

__version__ = "1.0.0a5"

from .client import AlphaClient
from .engine import (
    AssistantEvent,
    Engine,
    EngineState,
    ErrorEvent,
    Event,
    InitEvent,
    ResultEvent,
    StreamEvent,
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
    "StreamEvent",
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
