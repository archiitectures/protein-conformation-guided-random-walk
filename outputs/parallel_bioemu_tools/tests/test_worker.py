from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from parallel_bioemu_tools.manifests import PairRunSpec, WorkerConfig
from parallel_bioemu_tools.worker import (
    load_valid_completion_marker,
    marker_path_for_bundle,
    run_one_task,
    write_completion_marker,
)


class FakeStorage:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, str]] = []
        self.file_uploads: list[tuple[str, str]] = []
        self.remote_files: set[str] = set()
        self.remote_file_sources: dict[str, str] = {}

    def upload_bundle(self, local_dir, remote_prefix: str) -> str:
        self.uploads.append((str(local_dir), remote_prefix))
        root = local_dir.resolve()
        self.remote_files.update(
            f"{remote_prefix}/{path.relative_to(root).as_posix()}"
            for path in root.rglob("*")
            if path.is_file()
        )
        return f"{remote_prefix}/manifest.json"

    def upload_file(self, local_path, remote_path: str) -> str:
        self.file_uploads.append((str(local_path), remote_path))
        self.remote_files.add(remote_path)
        self.remote_file_sources[remote_path] = str(local_path)
        return remote_path

    def verify_remote_files(self, remote_paths: list[str]) -> None:
        missing = set(remote_paths) - self.remote_files
        if missing:
            raise RuntimeError(f"missing remote files: {missing}")

    def list_files(self) -> list[str]:
        return sorted(self.remote_files)

    def download_file(self, remote_path: str, local_dir) -> object:
        if remote_path not in self.remote_file_sources:
            raise FileNotFoundError(remote_path)
        local_dir = Path(local_dir).resolve()
        local_dir.mkdir(parents=True, exist_ok=True)
        source = self.remote_file_sources[remote_path]
        destination = local_dir / remote_path.split("/")[-1]
        shutil.copy2(source, destination)
        return destination


def _task() -> PairRunSpec:
    return PairRunSpec(
        protein_id="protein_1",
        pair_id="pair_x",
        fasta_path="/tmp/input.fasta",
        dataset_pair_dir="/tmp/pair",
        n_samples=32,
        estimated_cost=5.0,
    )


def _write_fake_bundle(output_dir, n_samples: int) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "nodes").mkdir(exist_ok=True)
    (output_dir / "nodes" / "sample_000.pdb").write_text("ATOM\n", encoding="utf-8")
    manifest = {
        "schema_version": "bioemu_bundle_v1",
        "sequence": {"id": "x", "sequence": "AAA", "length": 3},
        "n_samples_requested": n_samples,
        "samples": [{"pdb_path": "nodes/sample_000.pdb"}],
        "metadata": {},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest) + "\n",
        encoding="utf-8",
    )
    return manifest


def test_incomplete_completion_marker_is_ignored(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    marker_path = marker_path_for_bundle(bundle_dir)
    marker_path.write_text(json.dumps({"run_id": "r1"}) + "\n", encoding="utf-8")
    assert load_valid_completion_marker(marker_path) is None


def test_worker_resume_and_skip_behavior(monkeypatch, tmp_path) -> None:
    task = _task()
    storage = FakeStorage()
    calls: list[str] = []

    def fake_run_bioemu_from_fasta(fasta_path, output_dir, n_samples, seed=None, **kwargs):
        calls.append(str(output_dir))
        return _write_fake_bundle(output_dir, n_samples)

    monkeypatch.setattr("parallel_bioemu_tools.worker.run_bioemu_from_fasta", fake_run_bioemu_from_fasta)
    first = run_one_task(task, run_id="run1", shard_id="shard_000", storage=storage, config=WorkerConfig(workspace_root=tmp_path))
    second = run_one_task(task, run_id="run1", shard_id="shard_000", storage=storage, config=WorkerConfig(workspace_root=tmp_path))
    assert first == second
    assert len(calls) == 1
    assert len(storage.uploads) == 1
    assert len(storage.file_uploads) == 1

    storage.remote_files.clear()
    third = run_one_task(
        task,
        run_id="run1",
        shard_id="shard_000",
        storage=storage,
        config=WorkerConfig(workspace_root=tmp_path),
    )
    assert third == first
    assert len(calls) == 1
    assert len(storage.uploads) == 2
    assert len(storage.file_uploads) == 2


def test_worker_does_not_mark_completion_after_upload_failure(monkeypatch, tmp_path) -> None:
    task = _task()

    class FailingStorage(FakeStorage):
        def upload_bundle(self, local_dir, remote_prefix: str) -> str:
            raise RuntimeError("remote verification failed")

    monkeypatch.setattr(
        "parallel_bioemu_tools.worker.run_bioemu_from_fasta",
        lambda fasta_path, output_dir, n_samples, **kwargs: _write_fake_bundle(
            output_dir, n_samples
        ),
    )
    with pytest.raises(RuntimeError, match="remote verification failed"):
        run_one_task(
            task,
            run_id="run1",
            shard_id="shard_000",
            storage=FailingStorage(),
            config=WorkerConfig(workspace_root=tmp_path),
        )
    marker_path = tmp_path / "run1" / "protein_1" / "pair_x" / "n_32" / "_worker_completed.json"
    assert not marker_path.exists()


def test_worker_resumes_valid_bundle_without_resampling(monkeypatch, tmp_path) -> None:
    task = _task()
    storage = FakeStorage()
    bundle_dir = tmp_path / "run1" / "protein_1" / "pair_x" / "n_32"
    (bundle_dir / "nodes").mkdir(parents=True)
    (bundle_dir / "nodes" / "sample_000.pdb").write_text("ATOM\n", encoding="utf-8")
    (bundle_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "bioemu_bundle_v1",
                "samples": [{"pdb_path": "nodes/sample_000.pdb"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "parallel_bioemu_tools.worker.run_bioemu_from_fasta",
        lambda *args, **kwargs: pytest.fail("valid bundle should not be resampled"),
    )
    marker = run_one_task(
        task,
        run_id="run1",
        shard_id="shard_000",
        storage=storage,
        config=WorkerConfig(workspace_root=tmp_path),
    )
    assert marker.sample_count == 1
    assert len(storage.uploads) == 1


def test_worker_refuses_overwrite_without_marker(monkeypatch, tmp_path) -> None:
    task = _task()
    storage = FakeStorage()
    bundle_dir = tmp_path / "run1" / "protein_1" / "pair_x" / "n_32"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "manifest.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr("parallel_bioemu_tools.worker.run_bioemu_from_fasta", lambda *args, **kwargs: {})
    with pytest.raises(FileExistsError):
        run_one_task(task, run_id="run1", shard_id="shard_000", storage=storage, config=WorkerConfig(workspace_root=tmp_path))


def test_worker_uses_remote_completion_marker_on_new_runtime(monkeypatch, tmp_path) -> None:
    task = _task()
    storage = FakeStorage()
    bundle_dir = tmp_path / "run1" / "protein_1" / "pair_x" / "n_32"
    _write_fake_bundle(bundle_dir, n_samples=32)
    marker = run_one_task(
        task,
        run_id="run1",
        shard_id="shard_000",
        storage=storage,
        config=WorkerConfig(workspace_root=tmp_path),
    )

    new_workspace = tmp_path / "fresh-runtime"
    monkeypatch.setattr(
        "parallel_bioemu_tools.worker.run_bioemu_from_fasta",
        lambda *args, **kwargs: pytest.fail("remote marker should avoid resampling"),
    )
    resumed = run_one_task(
        task,
        run_id="run1",
        shard_id="shard_000",
        storage=storage,
        config=WorkerConfig(workspace_root=new_workspace),
    )
    assert resumed == marker
    local_marker = (
        new_workspace / "run1" / "protein_1" / "pair_x" / "n_32" / "_worker_completed.json"
    )
    assert load_valid_completion_marker(local_marker) == marker


def test_worker_does_not_resample_when_remote_lookup_fails(monkeypatch, tmp_path) -> None:
    task = _task()

    class FailingLookupStorage(FakeStorage):
        def list_files(self) -> list[str]:
            raise RuntimeError("Hugging Face is temporarily unavailable")

    monkeypatch.setattr(
        "parallel_bioemu_tools.worker.run_bioemu_from_fasta",
        lambda *args, **kwargs: pytest.fail("remote lookup failure must not start sampling"),
    )
    with pytest.raises(RuntimeError, match="temporarily unavailable"):
        run_one_task(
            task,
            run_id="run1",
            shard_id="shard_000",
            storage=FailingLookupStorage(),
            config=WorkerConfig(workspace_root=tmp_path),
        )


def test_worker_repairs_remote_outputs_when_local_marker_exists(tmp_path) -> None:
    task = _task()
    storage = FakeStorage()
    bundle_dir = tmp_path / "run1" / "protein_1" / "pair_x" / "n_32"
    _write_fake_bundle(bundle_dir, n_samples=32)
    marker = run_one_task(
        task,
        run_id="run1",
        shard_id="shard_000",
        storage=storage,
        config=WorkerConfig(workspace_root=tmp_path),
    )

    storage.remote_files.clear()
    write_completion_marker(bundle_dir / "_worker_completed.json", marker)
    resumed = run_one_task(
        task,
        run_id="run1",
        shard_id="shard_000",
        storage=storage,
        config=WorkerConfig(workspace_root=tmp_path),
    )
    assert resumed == marker
    assert marker.remote_manifest_path in storage.remote_files
    assert marker.remote_marker_path in storage.remote_files
