"""Thread-safe worker heartbeat renewal."""

from __future__ import annotations

import threading
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .hf_storage import HuggingFaceStorage
from .manifests import WorkerHeartbeat, write_json


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class HeartbeatController:
    """Renew a local and optional remote heartbeat from a daemon thread."""

    def __init__(self, heartbeat: WorkerHeartbeat, path: str | Path, *, interval_seconds: float = 120.0,
                 storage: HuggingFaceStorage | None = None, remote_path: str | None = None,
                 clock: Callable[[], str] = _now) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.heartbeat = heartbeat
        self.path = Path(path)
        self.interval_seconds = interval_seconds
        self.storage = storage
        self.remote_path = remote_path
        self.clock = clock
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: Exception | None = None

    def renew(self, **changes: object) -> WorkerHeartbeat:
        """Persist one heartbeat update; storage calls are serialized."""
        with self._lock:
            self.heartbeat = replace(self.heartbeat, updated_at_utc=self.clock(), **changes)
            write_json(self.path, asdict(self.heartbeat))
            if self.storage is not None and self.remote_path:
                self.storage.upload_file(self.path, self.remote_path)
            self.last_error = None
            return self.heartbeat

    def start(self) -> None:
        """Start the renewal daemon, if it is not already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self.renew()
        self._thread = threading.Thread(target=self._loop, name="bioemu-heartbeat", daemon=True)
        self._thread.start()

    def stop(self, *, state: str = "stopped") -> None:
        """Stop renewal and make a best-effort terminal-state update."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds + 1.0))
        try:
            self.renew(state=state, active_task=None, active_chunk=None)
        except Exception as exc:  # Monitoring failure must not hide task results.
            self.last_error = exc

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.renew()
            except Exception as exc:
                # A transient Hub error should not permanently kill renewal.
                self.last_error = exc


WorkerHeartbeatController = HeartbeatController
