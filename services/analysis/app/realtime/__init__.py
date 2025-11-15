"""
Real-time Updates Module

Provides WebSocket/SSE support for pushing live updates to clients:
- CVD run progress
- Thickness growth estimates
- Stress risk flags
- Adhesion risk indicators
"""

from .events import (
    emit_run_event,
    RunEventType,
    subscribe_to_run,
    get_run_events,
)

from .websocket import (
    WebSocketManager,
    get_ws_manager,
)

__all__ = [
    # Events
    "emit_run_event",
    "RunEventType",
    "subscribe_to_run",
    "get_run_events",

    # WebSocket
    "WebSocketManager",
    "get_ws_manager",
]
