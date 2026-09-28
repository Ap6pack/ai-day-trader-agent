#!/usr/bin/env python3
"""
In-process event bus for the live trading desk.

The analysis pipeline, trading workflow and Alpaca executor run in worker
threads, while the desk WebSocket lives on the asyncio loop. Publishers call
``publish()`` from any thread; subscribers receive events on an
``asyncio.Queue`` bound to their own loop. A bounded history lets a freshly
connected desk replay what happened before it opened.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import threading
from collections import deque
from datetime import date, datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

HISTORY_SIZE = 500
QUEUE_SIZE = 1000


def _jsonable(value: Any) -> Any:
    """Convert pipeline output (datetimes, numpy scalars, DataFrames) to JSON-safe values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable(item())
        except (TypeError, ValueError):
            pass
    return str(value)


class EventBus:
    """Thread-safe publish/subscribe hub with a replay buffer."""

    def __init__(self, history_size: int = HISTORY_SIZE) -> None:
        self._lock = threading.Lock()
        self._history: Deque[Dict[str, Any]] = deque(maxlen=history_size)
        self._subscribers: List[Tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = []
        self._ids = itertools.count(1)

    def publish(
        self,
        event_type: str,
        message: str = "",
        *,
        level: str = "info",
        symbol: Optional[str] = None,
        user_id: Optional[int] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record an event and fan it out to every subscriber. Never raises."""
        event = {
            "id": next(self._ids),
            "type": event_type,
            "level": level,
            "message": message,
            "symbol": symbol,
            "user_id": user_id,
            "data": _jsonable(data or {}),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            self._history.append(event)
            subscribers = list(self._subscribers)

        for loop, queue in subscribers:
            try:
                loop.call_soon_threadsafe(self._offer, queue, event)
            except RuntimeError:
                # Subscriber's loop is closed; it will be pruned on unsubscribe.
                pass
        return event

    @staticmethod
    def _offer(queue: asyncio.Queue, event: Dict[str, Any]) -> None:
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        queue.put_nowait(event)

    def subscribe(self) -> asyncio.Queue:
        """Register a subscriber on the running loop and return its queue."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_SIZE)
        loop = asyncio.get_running_loop()
        with self._lock:
            self._subscribers.append((loop, queue))
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        with self._lock:
            self._subscribers = [(l, q) for l, q in self._subscribers if q is not queue]

    def history(self, limit: int = 100, user_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return recent events visible to ``user_id`` (global events plus their own)."""
        with self._lock:
            events = list(self._history)
        visible = [e for e in events if e["user_id"] is None or e["user_id"] == user_id]
        return visible[-limit:] if limit else visible

    def clear(self) -> None:
        with self._lock:
            self._history.clear()


event_bus = EventBus()


def publish_event(event_type: str, message: str = "", **kwargs: Any) -> None:
    """Best-effort publish for instrumentation points; failures are only logged."""
    try:
        event_bus.publish(event_type, message, **kwargs)
    except Exception as exc:  # pragma: no cover - instrumentation must never break trading
        logger.debug(f"Event publish failed for {event_type}: {exc}")
