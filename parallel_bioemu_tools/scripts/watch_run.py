"""Inspect remote shard heartbeats from macOS and queue recovery records."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from parallel_bioemu_tools.hf_storage import HuggingFaceStorage
from parallel_bioemu_tools.manifests import RecoveryRecord, read_json, run_manifest_from_dict
from parallel_bioemu_tools.watchdog import MacWatchdog


def inspect_run(run_manifest: Path, *, storage: HuggingFaceStorage, now: float | None = None, stale_after: float = 900.0) -> dict:
    """Inspect every expected shard heartbeat under ``run_id/_heartbeats``."""
    observed_at = time.time() if now is None else now
    manifest = run_manifest_from_dict(read_json(run_manifest))
    prefix = f"{manifest.run_id}/_heartbeats/"
    remote_files = set(storage.list_files())
    heartbeat_paths = {
        shard.shard_id: f"{prefix}{shard.shard_id}.json" for shard in manifest.shards
    }
    completed_shards = {
        shard.shard_id
        for shard in manifest.shards
        if all(
            f"{manifest.run_id}/{task.protein_id}/{task.pair_id}/n_{task.n_samples}/"
            "_worker_completed.json" in remote_files
            for task in shard.tasks
        )
    }
    expected = {
        shard_id: path
        for shard_id, path in heartbeat_paths.items()
        if shard_id not in completed_shards
    }
    records: list[RecoveryRecord] = []
    with tempfile.TemporaryDirectory(prefix="parallel-bioemu-heartbeats-") as cache:
        local_paths = []
        remote_for_local = {}
        for shard_id, remote_path in expected.items():
            if remote_path not in remote_files:
                records.append(RecoveryRecord(
                    manifest.run_id, shard_id, "", "missing heartbeat",
                    datetime.fromtimestamp(observed_at, timezone.utc).isoformat(), remote_path,
                ))
                continue
            local_path = storage.download_file(remote_path, cache)
            local_paths.append(local_path)
            remote_for_local[str(local_path)] = (shard_id, remote_path)
        records.extend(MacWatchdog(stale_after_seconds=stale_after).inspect(local_paths, now=observed_at))
        normalized = []
        for record in records:
            if record.heartbeat_path in remote_for_local:
                shard_id, remote_path = remote_for_local[record.heartbeat_path]
                record = RecoveryRecord(record.run_id or manifest.run_id, record.shard_id or shard_id,
                                        record.worker_id, record.reason, record.observed_at_utc, remote_path,
                                        record.active_chunk)
            normalized.append(record)
    return {
        "run_id": manifest.run_id,
        "hf_repo_id": manifest.hf_repo_id,
        "checked_at_utc": datetime.fromtimestamp(observed_at, timezone.utc).isoformat(),
        "heartbeat_prefix": prefix,
        "expected_shards": sorted(heartbeat_paths),
        "completed_shards": sorted(completed_shards),
        "remote_heartbeat_files": sorted(remote_files & set(heartbeat_paths.values())),
        "recovery": [asdict(record) for record in normalized],
    }


def run_once(run_manifest: Path, status_path: Path, recovery_path: Path, *, repo_id: str | None = None,
             now: float | None = None, stale_after: float = 900.0) -> dict:
    payload = read_json(run_manifest)
    resolved_repo = repo_id or str(payload["hf_repo_id"])
    status = inspect_run(run_manifest, storage=HuggingFaceStorage(repo_id=resolved_repo),
                         now=now, stale_after=stale_after)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    recovery_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    recovery_path.write_text(json.dumps(status["recovery"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--recovery-queue", type=Path, required=True)
    parser.add_argument("--hf-repo-id", default=None)
    parser.add_argument("--poll-interval", type=float, default=300.0)
    parser.add_argument("--stale-after", type=float, default=900.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.poll_interval <= 0 or args.stale_after <= 0:
        raise SystemExit("poll and stale intervals must be positive")
    while True:
        status = run_once(args.run_manifest, args.status_path, args.recovery_queue,
                          repo_id=args.hf_repo_id, stale_after=args.stale_after)
        print(json.dumps({"run_id": status["run_id"], "recovery_records": len(status["recovery"])}), flush=True)
        if args.once:
            return
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
