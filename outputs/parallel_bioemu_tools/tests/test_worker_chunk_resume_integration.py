from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from parallel_bioemu_tools.manifests import (
    CompletionMarker,
    PairRunSpec,
    RunManifest,
    ShardAssignment,
    WorkerConfig,
    write_json,
)
from parallel_bioemu_tools.worker import run_one_task

from test_worker import FakeStorage, _write_fake_bundle


class ChunkResumeStorage(FakeStorage):
    def __init__(self) -> None:
        super().__init__()
        self.bundle_sources: dict[str, Path] = {}

    def upload_bundle(self, local_dir, remote_prefix: str) -> str:
        result = super().upload_bundle(local_dir, remote_prefix)
        self.bundle_sources[remote_prefix] = Path(local_dir).resolve()
        return result

    def download_prefix(self, remote_prefix: str, local_dir) -> Path:
        source = self.bundle_sources[remote_prefix]
        destination = Path(local_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for path in source.rglob("*"):
            if path.is_file():
                target = destination / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        return destination


def test_interrupted_chunks_resume_only_missing_chunk(monkeypatch, tmp_path):
    task = PairRunSpec("P1", "pair", "/tmp/a.fasta", "/tmp/pair", 45, 1.0, seed=3)
    storage = ChunkResumeStorage()
    calls: list[int] = []

    def interrupted_generator(fasta_path, output_dir, n_samples, seed=None, **kwargs):
        calls.append(n_samples)
        if n_samples == 5 and calls.count(5) == 1:
            raise RuntimeError("worker interrupted after chunks 0 and 1")
        return _write_fake_bundle(output_dir, n_samples)

    monkeypatch.setattr("parallel_bioemu_tools.worker.run_bioemu_from_fasta", interrupted_generator)
    with pytest.raises(RuntimeError, match="interrupted"):
        run_one_task(task, run_id="run1", shard_id="shard_000", storage=storage,
                     config=WorkerConfig(workspace_root=tmp_path / "first", chunk_size=20))

    resumed = run_one_task(task, run_id="run1", shard_id="shard_000", storage=storage,
                           config=WorkerConfig(workspace_root=tmp_path / "fresh", chunk_size=20))
    assert calls == [20, 20, 5, 5]
    assert isinstance(resumed, CompletionMarker)
    assert resumed.n_samples == 45
    assert resumed.sample_count == 3

    # The resilient contract requires the final root bundle and canonical marker
    # to remain visible to the existing collector boundary.
    root_prefix = "run1/P1/pair/n_45"
    assert f"{root_prefix}/manifest.json" in storage.remote_files
    assert f"{root_prefix}/_worker_completed.json" in storage.remote_files
    assert any(path.startswith("run1/_chunks/P1/pair/n_45/") for path in storage.remote_files)
    assert not any(path.startswith(f"{root_prefix}/chunks/") for path in storage.remote_files)


def test_worker_heartbeat_uses_watchdog_shard_path(monkeypatch, tmp_path):
    task = PairRunSpec("P1", "pair", "/tmp/a.fasta", "/tmp/pair", 2, 1.0)
    storage = ChunkResumeStorage()
    monkeypatch.setattr(
        "parallel_bioemu_tools.worker.run_bioemu_from_fasta",
        lambda fasta_path, output_dir, n_samples, **kwargs: _write_fake_bundle(
            output_dir, n_samples
        ),
    )

    run_one_task(
        task,
        run_id="run1",
        shard_id="shard_000",
        storage=storage,
        config=WorkerConfig(
            workspace_root=tmp_path,
            chunk_size=20,
            worker_id="worker-a",
            enable_heartbeat=True,
        ),
    )

    heartbeat_path = "run1/_heartbeats/shard_000.json"
    assert heartbeat_path in storage.remote_files
    heartbeat_source = Path(storage.remote_file_sources[heartbeat_path])
    payload = json.loads(heartbeat_source.read_text(encoding="utf-8"))
    assert payload["worker_id"] == "worker-a"
    assert payload["state"] == "completed"
