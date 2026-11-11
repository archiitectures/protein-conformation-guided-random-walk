"""Stage-level JSON checkpoints for resumable dataset construction."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


class StateStore:
    """Persist stage payloads under the dataset's hidden state directory."""

    def __init__(self, output_dir: str | Path, fingerprint_payload: dict[str, Any]) -> None:
        self.state_dir = Path(output_dir) / ".state"
        encoded = json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":"))
        self.fingerprint = hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def load(self, unit_id: str, stage: str, *, enabled: bool) -> Any | None:
        """Load a matching completed stage, or return ``None``."""
        if not enabled:
            return None
        path = self._path(unit_id, stage)
        if not path.exists():
            return None
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if envelope.get("fingerprint") != self.fingerprint or not envelope.get("completed"):
            return None
        return envelope.get("payload")

    def save(self, unit_id: str, stage: str, payload: Any) -> Path:
        """Atomically mark a stage complete and store its payload."""
        path = self._path(unit_id, stage)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        envelope = {"fingerprint": self.fingerprint, "completed": True, "payload": payload}
        temporary.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        return path

    def _path(self, unit_id: str, stage: str) -> Path:
        safe_unit = re.sub(r"[^A-Za-z0-9_.-]+", "_", unit_id)
        return self.state_dir / safe_unit / f"{stage}.json"
