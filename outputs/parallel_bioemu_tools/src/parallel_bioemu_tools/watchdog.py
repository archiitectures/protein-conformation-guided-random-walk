"""Mac-side watchdog that only observes workers and queues recovery."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .manifests import RecoveryRecord


class MacWatchdog:
    """Inspect heartbeat files; it never imports or invokes BioEmu or GPU code."""

    def __init__(self, *, stale_after_seconds: float = 15 * 60) -> None:
        if stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be positive")
        self.stale_after_seconds = stale_after_seconds

    def inspect(self, heartbeat_paths: Iterable[str | Path], *, now: float | None = None) -> list[RecoveryRecord]:
        """Return recovery records for missing, malformed, or stale heartbeats."""
        current = now if now is not None else datetime.now(timezone.utc).timestamp()
        records = []
        for value in heartbeat_paths:
            path = Path(value)
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                state = str(data.get("state", ""))
                if state == "completed":
                    continue
                updated = datetime.fromisoformat(str(data["updated_at_utc"])).timestamp()
                age = current - updated
                if state == "failed":
                    reason = "worker reported failed"
                elif age <= self.stale_after_seconds:
                    continue
                else:
                    reason = f"heartbeat stale for {age:.0f}s"
                records.append(RecoveryRecord(str(data.get("run_id", "")), str(data.get("shard_id", "")),
                    str(data.get("worker_id", "")), reason, datetime.fromtimestamp(current, timezone.utc).isoformat(),
                    str(path), data.get("active_chunk")))
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                records.append(RecoveryRecord("", "", "", f"invalid heartbeat: {type(exc).__name__}",
                    datetime.fromtimestamp(current, timezone.utc).isoformat(), str(path)))
        return records

    def write_recovery_queue(self, records: Iterable[RecoveryRecord], path: str | Path) -> Path:
        """Write recovery records for an external supervisor to consume."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps([asdict(record) for record in records], indent=2) + "\n", encoding="utf-8")
        return output
