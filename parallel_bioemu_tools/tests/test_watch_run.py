from __future__ import annotations

import importlib.util
import inspect
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from parallel_bioemu_tools.manifests import PairRunSpec, RunManifest, ShardAssignment, write_json


def load_watchdog():
    script = Path(__file__).resolve().parents[1] / "scripts/watch_run.py"
    spec = importlib.util.spec_from_file_location("watch_run", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RemoteHeartbeatStorage:
    def __init__(self, files: dict[str, dict]) -> None:
        self.files = files

    def list_files(self) -> list[str]:
        return sorted(self.files)

    def download_file(self, remote_path: str, local_dir: str | Path) -> Path:
        destination = Path(local_dir) / Path(remote_path).name
        destination.write_text(json.dumps(self.files[remote_path]), encoding="utf-8")
        return destination


def manifest(tmp_path):
    task = PairRunSpec("P1", "pair", "/tmp/a.fasta", "/tmp/pair", 45, 1.0)
    shards = tuple(
        ShardAssignment("run1", f"shard_{index:03d}", index, 3, (task,), 1.0)
        for index in range(3)
    )
    value = RunManifest("run1", "org/repo", (task,), shards)
    path = tmp_path / "run.json"
    write_json(path, asdict(value))
    return path


def test_remote_fresh_stale_missing_and_invalid_shards(tmp_path):
    watch = load_watchdog()
    now = 1_700_000_000.0
    fresh = datetime.fromtimestamp(now - 10, timezone.utc).isoformat()
    stale = datetime.fromtimestamp(now - 901, timezone.utc).isoformat()
    storage = RemoteHeartbeatStorage({
        "run1/_heartbeats/shard_000.json": {"run_id": "run1", "shard_id": "shard_000", "worker_id": "w0", "updated_at_utc": fresh},
        "run1/_heartbeats/shard_001.json": {"run_id": "run1", "shard_id": "shard_001", "worker_id": "w1", "updated_at_utc": stale},
        "run1/_heartbeats/shard_002.json": {"not": "a heartbeat"},
    })
    status = watch.inspect_run(manifest(tmp_path), storage=storage, now=now)
    reasons = {record["shard_id"]: record["reason"] for record in status["recovery"]}
    assert reasons["shard_001"].startswith("heartbeat stale")
    assert reasons["shard_002"].startswith("invalid heartbeat")
    assert reasons["shard_002"]
    assert reasons.get("shard_000") is None


def test_terminal_and_durable_completion_do_not_trigger_recovery(tmp_path):
    watch = load_watchdog()
    now = 1_700_000_000.0
    stale = datetime.fromtimestamp(now - 10_000, timezone.utc).isoformat()
    storage = RemoteHeartbeatStorage({
        "run1/_heartbeats/shard_000.json": {
            "run_id": "run1", "shard_id": "shard_000", "worker_id": "w0",
            "state": "completed", "updated_at_utc": stale,
        },
        "run1/_heartbeats/shard_001.json": {
            "run_id": "run1", "shard_id": "shard_001", "worker_id": "w1",
            "state": "failed", "updated_at_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
        },
        "run1/shard-marker-placeholder": {},
    })
    status = watch.inspect_run(manifest(tmp_path), storage=storage, now=now)
    reasons = {record["shard_id"]: record["reason"] for record in status["recovery"]}
    assert "shard_000" not in reasons
    assert reasons["shard_001"] == "worker reported failed"

    completion_path = "run1/P1/pair/n_45/_worker_completed.json"
    storage.files[completion_path] = {}
    completed = watch.inspect_run(manifest(tmp_path), storage=storage, now=now)
    assert completed["completed_shards"] == ["shard_000", "shard_001", "shard_002"]
    assert completed["recovery"] == []


def test_once_writes_status_and_recovery_and_uses_hf_token_behavior(tmp_path, monkeypatch, capsys):
    watch = load_watchdog()
    storage = RemoteHeartbeatStorage({})
    monkeypatch.setattr(watch, "HuggingFaceStorage", lambda repo_id: storage)
    run_manifest = manifest(tmp_path)
    status_path, recovery_path = tmp_path / "status.json", tmp_path / "recovery.json"
    monkeypatch.setattr(watch, "run_once", lambda *args, **kwargs: {
        "run_id": "run1", "recovery": [{"shard_id": "shard_000"}],
    })
    monkeypatch.setattr(watch, "parse_args", lambda: type("Args", (), {
        "run_manifest": run_manifest, "status_path": status_path, "recovery_queue": recovery_path,
        "hf_repo_id": None, "poll_interval": 300.0, "stale_after": 900.0, "once": True,
    })())
    watch.main()
    assert "recovery_records" in capsys.readouterr().out


def test_watchdog_has_no_compute_imports():
    watch = load_watchdog()
    source = inspect.getsource(watch)
    assert "colab_gpu" not in source
    assert "parallel_bioemu_tools.worker" not in source
    assert "run_bioemu" not in source
