"""Append-only audit events for proposal-to-outcome traceability."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Dict, List, Optional

from .models import AuditEvent, TransactionState


class AuditLog:
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = Path(path) if path else None
        self._events: List[AuditEvent] = []
        self._lock = threading.RLock()

    def append(self, event: AuditEvent) -> None:
        with self._lock:
            self._events.append(event)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                line = event.model_dump_json(exclude_none=True) + "\n"
                fd = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
                try:
                    os.write(fd, line.encode("utf-8"))
                finally:
                    os.close(fd)

    def events(self, transaction_id: Optional[str] = None) -> List[AuditEvent]:
        with self._lock:
            if transaction_id is None:
                return list(self._events)
            return [event for event in self._events if event.transaction_id == transaction_id]

    def state_counts(self, transaction_id: str) -> Dict[TransactionState, int]:
        counts: Dict[TransactionState, int] = {}
        for event in self.events(transaction_id):
            counts[event.state] = counts.get(event.state, 0) + 1
        return counts

    def has_validation_before_commit(self, transaction_id: str) -> bool:
        validation_seen = False
        for event in self.events(transaction_id):
            if event.payload.get("validation_safe") is True:
                validation_seen = True
            if event.state == TransactionState.COMMITTED:
                return validation_seen
        return False
