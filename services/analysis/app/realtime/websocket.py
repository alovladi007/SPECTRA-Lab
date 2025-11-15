"""
WebSocket Manager

Manages WebSocket connections for real-time updates.
Integrates with FastAPI WebSocket support.
"""

import asyncio
import logging
from typing import Dict, Set, Optional
from dataclasses import dataclass
import json

try:
    from fastapi import WebSocket, WebSocketDisconnect
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    WebSocket = None
    WebSocketDisconnect = Exception

from .events import get_event_publisher, subscribe_to_run

logger = logging.getLogger(__name__)


@dataclass
class ConnectionInfo:
    """Information about a WebSocket connection"""
    websocket: WebSocket
    run_id: str
    client_id: str
    connected_at: str


class WebSocketManager:
    """
    Manages WebSocket connections for real-time CVD run updates

    Keeps track of active connections and broadcasts events to subscribers.
    """

    def __init__(self):
        # Map of run_id -> set of websockets
        self.active_connections: Dict[str, Set[WebSocket]] = {}

        # Map of websocket -> connection info
        self.connection_info: Dict[WebSocket, ConnectionInfo] = {}

    async def connect(
        self,
        websocket: WebSocket,
        run_id: str,
        client_id: str = "unknown",
    ):
        """
        Accept a new WebSocket connection

        Args:
            websocket: WebSocket connection
            run_id: Run ID to subscribe to
            client_id: Client identifier
        """
        await websocket.accept()

        # Add to active connections
        if run_id not in self.active_connections:
            self.active_connections[run_id] = set()

        self.active_connections[run_id].add(websocket)

        # Store connection info
        from datetime import datetime
        self.connection_info[websocket] = ConnectionInfo(
            websocket=websocket,
            run_id=run_id,
            client_id=client_id,
            connected_at=datetime.now().isoformat(),
        )

        logger.info(f"WebSocket connected: client={client_id}, run_id={run_id}")

        # Send welcome message
        await self.send_message(
            websocket,
            {
                "type": "connected",
                "run_id": run_id,
                "message": f"Connected to run {run_id}",
            }
        )

    def disconnect(self, websocket: WebSocket):
        """
        Disconnect a WebSocket

        Args:
            websocket: WebSocket to disconnect
        """
        # Get connection info
        info = self.connection_info.get(websocket)

        if info:
            # Remove from active connections
            if info.run_id in self.active_connections:
                self.active_connections[info.run_id].discard(websocket)

                # Clean up empty sets
                if not self.active_connections[info.run_id]:
                    del self.active_connections[info.run_id]

            # Remove connection info
            del self.connection_info[websocket]

            logger.info(f"WebSocket disconnected: client={info.client_id}, run_id={info.run_id}")

    async def send_message(
        self,
        websocket: WebSocket,
        message: dict,
    ):
        """
        Send a message to a specific WebSocket

        Args:
            websocket: Target websocket
            message: Message dictionary
        """
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            self.disconnect(websocket)

    async def broadcast_to_run(
        self,
        run_id: str,
        message: dict,
    ):
        """
        Broadcast a message to all connections for a specific run

        Args:
            run_id: Run identifier
            message: Message dictionary
        """
        if run_id not in self.active_connections:
            return

        # Send to all connections
        disconnected = []

        for websocket in self.active_connections[run_id]:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Failed to broadcast: {e}")
                disconnected.append(websocket)

        # Clean up disconnected
        for ws in disconnected:
            self.disconnect(ws)

    async def stream_run_events(
        self,
        websocket: WebSocket,
        run_id: str,
    ):
        """
        Stream events for a run to a WebSocket connection

        This runs in a background task and pushes events as they arrive.

        Args:
            websocket: WebSocket connection
            run_id: Run identifier
        """
        logger.info(f"Starting event stream for run_id={run_id}")

        try:
            # Subscribe to run events
            for event in subscribe_to_run(run_id):
                # Check if still connected
                if websocket not in self.connection_info:
                    break

                # Send event to client
                await self.send_message(
                    websocket,
                    {
                        "type": "event",
                        "event_type": event.event_type.value,
                        "timestamp": event.timestamp,
                        "data": event.data,
                    }
                )

                # Stop if run completed or failed
                if event.event_type in ["run_completed", "run_failed", "run_cancelled"]:
                    break

        except Exception as e:
            logger.error(f"Error in event stream: {e}")

        finally:
            logger.info(f"Event stream ended for run_id={run_id}")

    def get_active_connections_count(self, run_id: Optional[str] = None) -> int:
        """
        Get count of active connections

        Args:
            run_id: Optional run ID to filter by

        Returns:
            Number of active connections
        """
        if run_id:
            return len(self.active_connections.get(run_id, set()))
        else:
            return sum(len(conns) for conns in self.active_connections.values())


# Global WebSocket manager
_ws_manager: Optional[WebSocketManager] = None


def get_ws_manager() -> WebSocketManager:
    """Get or create global WebSocket manager"""
    global _ws_manager

    if _ws_manager is None:
        _ws_manager = WebSocketManager()

    return _ws_manager
