"""
Real-time Event System

Publishes events to Redis for real-time updates to connected clients.
"""

import json
import logging
from enum import Enum
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class RunEventType(str, Enum):
    """Types of run events"""
    RUN_STARTED = "run_started"
    PROGRESS_UPDATE = "progress_update"
    METRICS_UPDATE = "metrics_update"
    WARNING = "warning"
    ERROR = "error"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    RUN_CANCELLED = "run_cancelled"

    # Specific metric events
    THICKNESS_UPDATE = "thickness_update"
    STRESS_RISK = "stress_risk"
    ADHESION_RISK = "adhesion_risk"
    RATE_ANOMALY = "rate_anomaly"


@dataclass
class RunEvent:
    """Real-time run event"""
    run_id: str
    event_type: RunEventType
    timestamp: str
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "run_id": self.run_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "data": self.data,
        }


class EventPublisher:
    """
    Publishes events to Redis Pub/Sub for real-time delivery

    Events are published to Redis channels that WebSocket/SSE servers subscribe to.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """
        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url
        self._redis_client: Optional[redis.Redis] = None

        if REDIS_AVAILABLE:
            try:
                self._redis_client = redis.from_url(redis_url, decode_responses=True)
                logger.info(f"EventPublisher connected to Redis: {redis_url}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Events will be logged only.")
                self._redis_client = None
        else:
            logger.warning("Redis not available. Install: pip install redis")

    def publish(self, event: RunEvent):
        """
        Publish event to Redis channel

        Args:
            event: Event to publish
        """
        # Channel format: "cvd:run:{run_id}"
        channel = f"cvd:run:{event.run_id}"

        # Serialize event
        event_json = json.dumps(event.to_dict())

        # Publish to Redis
        if self._redis_client:
            try:
                self._redis_client.publish(channel, event_json)
                logger.debug(f"Published event to {channel}: {event.event_type.value}")
            except Exception as e:
                logger.error(f"Failed to publish event: {e}")
        else:
            # Fallback: just log
            logger.info(f"Event [{event.event_type.value}] for {event.run_id}: {event.data}")

    def store_event(self, event: RunEvent, ttl_seconds: int = 3600):
        """
        Store event in Redis list for history

        Args:
            event: Event to store
            ttl_seconds: Time to live for event history
        """
        # List key format: "cvd:events:{run_id}"
        list_key = f"cvd:events:{event.run_id}"

        if self._redis_client:
            try:
                # Push event to list
                self._redis_client.rpush(list_key, json.dumps(event.to_dict()))

                # Set expiration
                self._redis_client.expire(list_key, ttl_seconds)

                logger.debug(f"Stored event in {list_key}")
            except Exception as e:
                logger.error(f"Failed to store event: {e}")


# Global publisher instance
_event_publisher: Optional[EventPublisher] = None


def get_event_publisher() -> EventPublisher:
    """Get or create global event publisher"""
    global _event_publisher

    if _event_publisher is None:
        _event_publisher = EventPublisher()

    return _event_publisher


def emit_run_event(
    run_id: str,
    event_type: RunEventType,
    data: Dict[str, Any],
):
    """
    Emit a run event to all subscribers

    Args:
        run_id: Run identifier
        event_type: Type of event
        data: Event data
    """
    event = RunEvent(
        run_id=run_id,
        event_type=event_type,
        timestamp=datetime.now().isoformat(),
        data=data,
    )

    publisher = get_event_publisher()

    # Publish event (real-time)
    publisher.publish(event)

    # Store event (history)
    publisher.store_event(event)


def subscribe_to_run(run_id: str):
    """
    Subscribe to events for a specific run

    Returns a generator that yields events as they arrive.

    Args:
        run_id: Run identifier

    Yields:
        RunEvent objects
    """
    publisher = get_event_publisher()

    if not publisher._redis_client:
        logger.warning("Cannot subscribe without Redis connection")
        return

    # Channel to subscribe to
    channel = f"cvd:run:{run_id}"

    # Create pubsub
    pubsub = publisher._redis_client.pubsub()
    pubsub.subscribe(channel)

    logger.info(f"Subscribed to {channel}")

    try:
        for message in pubsub.listen():
            if message["type"] == "message":
                # Parse event
                event_dict = json.loads(message["data"])
                event = RunEvent(
                    run_id=event_dict["run_id"],
                    event_type=RunEventType(event_dict["event_type"]),
                    timestamp=event_dict["timestamp"],
                    data=event_dict["data"],
                )

                yield event

    except GeneratorExit:
        pubsub.unsubscribe(channel)
        logger.info(f"Unsubscribed from {channel}")


def get_run_events(
    run_id: str,
    limit: Optional[int] = None,
) -> List[RunEvent]:
    """
    Get event history for a run

    Args:
        run_id: Run identifier
        limit: Maximum number of events to return (most recent)

    Returns:
        List of RunEvent objects
    """
    publisher = get_event_publisher()

    if not publisher._redis_client:
        return []

    list_key = f"cvd:events:{run_id}"

    try:
        # Get events from Redis list
        if limit:
            event_jsons = publisher._redis_client.lrange(list_key, -limit, -1)
        else:
            event_jsons = publisher._redis_client.lrange(list_key, 0, -1)

        # Parse events
        events = []
        for event_json in event_jsons:
            event_dict = json.loads(event_json)
            events.append(RunEvent(
                run_id=event_dict["run_id"],
                event_type=RunEventType(event_dict["event_type"]),
                timestamp=event_dict["timestamp"],
                data=event_dict["data"],
            ))

        return events

    except Exception as e:
        logger.error(f"Failed to get run events: {e}")
        return []
