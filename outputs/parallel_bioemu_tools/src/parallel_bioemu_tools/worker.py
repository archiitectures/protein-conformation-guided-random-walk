"""Explicit shard worker for BioEmu bundle generation."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from guided_conformational_path.bioemu_io import load_bioemu_manifest
from guided_conformational_path.bioemu_runner import run_bioemu_from_fasta

from .hf_storage import HuggingFaceStorage
from .heartbeat import HeartbeatController
from .merge import merge_chunk_bundles
from .manifests import (
    ChunkCompletionMarker,
    CompletionMarker,
    PairRunSpec,
    SampleChunkSpec,
    WorkerHeartbeat,
    WorkerConfig,
    build_scoped_output_path,
    build_chunk_output_path,
    expand_pair_run_spec,
    read_json,
    shard_assignment_from_dict,
    write_json,
)


def marker_path_for_bundle(bundle_dir: str | Path) -> Path:
    """Return the local completion marker path for one bundle."""
    return Path(bundle_dir) / "_worker_completed.json"


def chunk_marker_path_for_bundle(bundle_dir: str | Path) -> Path:
    """Return the local completion marker path for one chunk."""
    return Path(bundle_dir) / "_chunk_completed.json"


def load_valid_completion_marker(path: str | Path) -> CompletionMarker | None:
    """Return a parsed marker only when it is complete and valid."""
    marker_file = Path(path)
    if not marker_file.is_file():
        return None
    try:
        data = json.loads(marker_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    required = {
        "run_id",
        "shard_id",
        "protein_id",
        "pair_id",
        "n_samples",
        "bundle_path",
        "remote_manifest_path",
        "remote_marker_path",
        "sample_count",
        "completed_at_utc",
    }
    if not required.issubset(data):
        return None
    marker = CompletionMarker(
        run_id=str(data["run_id"]),
        shard_id=str(data["shard_id"]),
        protein_id=str(data["protein_id"]),
        pair_id=str(data["pair_id"]),
        n_samples=int(data["n_samples"]),
        bundle_path=str(data["bundle_path"]),
        remote_manifest_path=str(data["remote_manifest_path"]),
        remote_marker_path=str(data["remote_marker_path"]),
        sample_count=int(data["sample_count"]),
        completed_at_utc=str(data["completed_at_utc"]),
    )
    if marker.n_samples <= 0 or marker.sample_count <= 0:
        return None
    return marker


def write_completion_marker(path: str | Path, marker: CompletionMarker) -> Path:
    """Write the completion marker atomically."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output.with_suffix(output.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(asdict(marker), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(output)
    return output


def load_valid_chunk_completion_marker(path: str | Path) -> ChunkCompletionMarker | None:
    """Return a parsed chunk marker only when its required fields are valid."""
    marker_file = Path(path)
    try:
        data = json.loads(marker_file.read_text(encoding="utf-8"))
        marker = ChunkCompletionMarker(**data)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if marker.total_samples <= 0 or marker.chunk_n_samples <= 0 or marker.sample_count <= 0:
        return None
    return marker


def write_chunk_completion_marker(path: str | Path, marker: ChunkCompletionMarker) -> Path:
    """Write a chunk marker atomically."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(asdict(marker), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return output


def run_worker_shard(
    shard_manifest_path: str | Path,
    *,
    storage: HuggingFaceStorage,
    config: WorkerConfig,
) -> list[CompletionMarker]:
    """Run all tasks in one immutable shard manifest."""
    shard = shard_assignment_from_dict(read_json(shard_manifest_path))
    heartbeat = _make_heartbeat(shard.run_id, shard.shard_id, storage=storage, config=config)
    if heartbeat is not None:
        heartbeat.start()
    try:
        completions: list[CompletionMarker] = []
        for task in shard.tasks:
            if heartbeat is not None:
                heartbeat.renew(active_task=f"{task.protein_id}/{task.pair_id}", active_chunk=None)
            completions.append(run_one_task(task, run_id=shard.run_id, shard_id=shard.shard_id,
                                            storage=storage, config=config, _heartbeat=heartbeat))
        if heartbeat is not None:
            heartbeat.stop(state="completed")
        return completions
    except Exception:
        if heartbeat is not None:
            heartbeat.stop(state="failed")
        raise


def run_one_task(
    task: PairRunSpec,
    *,
    run_id: str,
    shard_id: str,
    storage: HuggingFaceStorage,
    config: WorkerConfig,
    _heartbeat: HeartbeatController | None = None,
) -> CompletionMarker:
    """Generate, validate, upload, and mark one BioEmu bundle."""
    owned_heartbeat = _heartbeat
    if owned_heartbeat is None and config.enable_heartbeat:
        owned_heartbeat = _make_heartbeat(run_id, shard_id, storage=storage, config=config)
        assert owned_heartbeat is not None
        owned_heartbeat.start()
    try:
        return _run_one_task_with_heartbeat(task, run_id=run_id, shard_id=shard_id,
                                            storage=storage, config=config, heartbeat=owned_heartbeat)
    except Exception:
        if _heartbeat is None and owned_heartbeat is not None:
            owned_heartbeat.stop(state="failed")
        raise
    finally:
        if _heartbeat is None and owned_heartbeat is not None and owned_heartbeat.heartbeat.state != "failed":
            owned_heartbeat.stop(state="completed")


def _run_one_task_with_heartbeat(
    task: PairRunSpec, *, run_id: str, shard_id: str, storage: HuggingFaceStorage,
    config: WorkerConfig, heartbeat: HeartbeatController | None,
) -> CompletionMarker:
    """Dispatch one task while sharing the shard heartbeat."""
    if config.chunk_size is not None:
        return run_chunked_task(task, run_id=run_id, shard_id=shard_id, storage=storage,
                                config=config, heartbeat=heartbeat)
    return _run_legacy_task(task, run_id=run_id, shard_id=shard_id, storage=storage, config=config)


def run_chunked_task(
    task: PairRunSpec, *, run_id: str, shard_id: str, storage: HuggingFaceStorage,
    config: WorkerConfig, heartbeat: HeartbeatController | None = None,
) -> CompletionMarker:
    """Run chunks, assemble a canonical bundle, and return its standard marker."""
    canonical_dir = build_scoped_output_path(config.workspace_root, run_id=run_id,
        protein_id=task.protein_id, pair_id=task.pair_id, n_samples=task.n_samples)
    canonical_remote = f"{run_id}/{task.protein_id}/{task.pair_id}/n_{task.n_samples}"
    canonical_marker_path = marker_path_for_bundle(canonical_dir)
    existing = load_valid_completion_marker(canonical_marker_path)
    if existing is not None and not config.overwrite:
        _validate_marker_identity(existing, task, run_id=run_id, shard_id=shard_id)
        _verify_remote_completion(existing, storage)
        return existing
    if existing is None and not config.overwrite:
        remote_existing = _download_remote_completion_marker(
            storage, remote_marker_path=f"{canonical_remote}/_worker_completed.json"
        )
        if remote_existing is not None:
            _validate_marker_identity(remote_existing, task, run_id=run_id, shard_id=shard_id)
            _verify_remote_completion(remote_existing, storage)
            if canonical_dir.exists():
                shutil.rmtree(canonical_dir)
            storage.download_prefix(canonical_remote, canonical_dir)
            load_bioemu_manifest(canonical_dir)
            write_completion_marker(canonical_marker_path, remote_existing)
            return remote_existing
    markers: list[ChunkCompletionMarker] = []
    for chunk in expand_pair_run_spec(task, config.chunk_size or 20):
        if heartbeat is not None:
            heartbeat.renew(active_task=f"{task.protein_id}/{task.pair_id}", active_chunk=chunk.chunk_index)
        markers.append(_run_one_chunk(chunk, run_id=run_id, shard_id=shard_id, storage=storage, config=config))
    config.workspace_root.mkdir(parents=True, exist_ok=True)
    merged_temp = Path(tempfile.mkdtemp(prefix="parallel-bioemu-merged-", dir=config.workspace_root))
    try:
        merge_chunk_bundles((marker.bundle_path for marker in markers), merged_temp)
        canonical_dir.parent.mkdir(parents=True, exist_ok=True)
        _replace_directory(canonical_dir, merged_temp)
        merged_temp = canonical_dir
        remote_manifest_path = storage.upload_bundle(canonical_dir, canonical_remote)
        storage.verify_remote_files([remote_manifest_path])
        manifest = load_bioemu_manifest(canonical_dir)
        marker = CompletionMarker(run_id, shard_id, task.protein_id, task.pair_id, task.n_samples,
            str(canonical_dir), remote_manifest_path, f"{canonical_remote}/_worker_completed.json",
            len(manifest["samples"]), datetime.now(timezone.utc).isoformat())
        write_completion_marker(canonical_marker_path, marker)
        storage.upload_file(canonical_marker_path, marker.remote_marker_path)
        storage.verify_remote_files([remote_manifest_path, marker.remote_marker_path])
        return marker
    finally:
        if merged_temp.exists() and merged_temp != canonical_dir:
            shutil.rmtree(merged_temp)


def _run_one_chunk(
    chunk: SampleChunkSpec, *, run_id: str, shard_id: str,
    storage: HuggingFaceStorage, config: WorkerConfig,
) -> ChunkCompletionMarker:
    """Run one chunk, with its own immutable prefix and completion marker."""
    bundle_dir = build_chunk_output_path(config.workspace_root, run_id=run_id,
        protein_id=chunk.task.protein_id, pair_id=chunk.task.pair_id,
        total_samples=chunk.total_samples, chunk_index=chunk.chunk_index, chunk_size=chunk.n_samples)
    # Keep durable raw chunks outside the canonical bundle prefix. Otherwise a
    # collector download of ``n_TOTAL`` would recursively fetch every raw chunk.
    remote_prefix = f"{run_id}/_chunks/{chunk.storage_key}"
    marker_path = chunk_marker_path_for_bundle(bundle_dir)
    remote_marker_path = f"{remote_prefix}/_chunk_completed.json"
    if not config.overwrite:
        remote = _download_remote_chunk_marker(storage, remote_marker_path)
        if remote is not None:
            _validate_chunk_marker(remote, chunk, run_id=run_id, shard_id=shard_id)
            _verify_chunk_completion(remote, storage)
            remote = _ensure_local_chunk_bundle(remote, remote_prefix, bundle_dir, storage)
            write_chunk_completion_marker(marker_path, remote)
            return remote
    if bundle_dir.exists() and config.overwrite:
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    if not (bundle_dir / "manifest.json").is_file():
        run_bioemu_from_fasta(chunk.task.fasta_path, bundle_dir, n_samples=chunk.n_samples,
                              seed=chunk.seed, **(chunk.task.bioemu_kwargs or {}))
    manifest = load_bioemu_manifest(bundle_dir)
    _validate_sample_files(bundle_dir, manifest)
    remote_manifest_path = storage.upload_bundle(bundle_dir, remote_prefix)
    storage.verify_remote_files([remote_manifest_path])
    marker = ChunkCompletionMarker(run_id, shard_id, chunk.task.protein_id, chunk.task.pair_id,
        chunk.total_samples, chunk.chunk_index, chunk.n_samples, str(bundle_dir), remote_manifest_path,
        remote_marker_path, len(manifest["samples"]), datetime.now(timezone.utc).isoformat())
    write_chunk_completion_marker(marker_path, marker)
    storage.upload_file(marker_path, remote_marker_path)
    storage.verify_remote_files([remote_manifest_path, remote_marker_path])
    return marker


def _run_legacy_task(
    task: PairRunSpec,
    *,
    run_id: str,
    shard_id: str,
    storage: HuggingFaceStorage,
    config: WorkerConfig,
) -> CompletionMarker:
    """Original unchunked implementation retained for compatibility."""
    bundle_dir = build_scoped_output_path(
        config.workspace_root,
        run_id=run_id,
        protein_id=task.protein_id,
        pair_id=task.pair_id,
        n_samples=task.n_samples,
    )
    marker_path = marker_path_for_bundle(bundle_dir)
    existing = load_valid_completion_marker(marker_path)
    remote_prefix = f"{run_id}/{task.protein_id}/{task.pair_id}/n_{task.n_samples}"
    remote_marker_path = f"{remote_prefix}/_worker_completed.json"
    if existing is not None and not config.overwrite:
        _validate_marker_identity(existing, task, run_id=run_id, shard_id=shard_id)
        _ensure_remote_completion(bundle_dir, marker_path, existing, storage, remote_prefix)
        return existing
    if existing is None and not config.overwrite:
        remote_existing = _download_remote_completion_marker(
            storage,
            remote_marker_path=remote_marker_path,
        )
        if remote_existing is not None:
            _validate_marker_identity(remote_existing, task, run_id=run_id, shard_id=shard_id)
            _verify_remote_completion(remote_existing, storage)
            write_completion_marker(marker_path, remote_existing)
            return remote_existing
    if bundle_dir.exists() and config.overwrite:
        shutil.rmtree(bundle_dir)
    if bundle_dir.exists() and not config.overwrite and not _is_resumable(bundle_dir):
        raise FileExistsError(f"Refusing to overwrite invalid bundle directory: {bundle_dir}")

    bundle_dir.mkdir(parents=True, exist_ok=True)
    if not (bundle_dir / "manifest.json").is_file():
        run_bioemu_from_fasta(
            task.fasta_path,
            bundle_dir,
            n_samples=task.n_samples,
            seed=task.seed,
            **(task.bioemu_kwargs or {}),
        )
    validated = load_bioemu_manifest(bundle_dir)
    _validate_sample_files(bundle_dir, validated)
    sample_count = len(validated["samples"])
    remote_manifest_path = storage.upload_bundle(bundle_dir, remote_prefix)
    marker = CompletionMarker(
        run_id=run_id,
        shard_id=shard_id,
        protein_id=task.protein_id,
        pair_id=task.pair_id,
        n_samples=task.n_samples,
        bundle_path=str(bundle_dir),
        remote_manifest_path=remote_manifest_path,
        remote_marker_path=remote_marker_path,
        sample_count=sample_count,
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    write_completion_marker(marker_path, marker)
    storage.upload_file(marker_path, remote_marker_path)
    return marker


def _is_resumable(bundle_dir: Path) -> bool:
    """Return whether an incomplete directory can safely resume."""
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        return True
    try:
        manifest = load_bioemu_manifest(bundle_dir)
        _validate_sample_files(bundle_dir, manifest)
    except (FileNotFoundError, TypeError, ValueError):
        return False
    return True


def _validate_sample_files(bundle_dir: Path, manifest: dict[str, Any]) -> None:
    """Validate that a generated manifest has samples with safe existing PDB paths."""
    samples = manifest.get("samples", [])
    if not samples:
        raise ValueError(f"BioEmu bundle contains no samples: {bundle_dir}")
    root = bundle_dir.resolve()
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or not isinstance(sample.get("pdb_path"), str):
            raise ValueError(f"BioEmu sample {index} has no PDB path")
        candidate = (root / sample["pdb_path"]).resolve()
        if root not in candidate.parents or not candidate.is_file():
            raise ValueError(f"BioEmu sample {index} has an unsafe or missing PDB: {candidate}")


def _validate_marker_identity(
    marker: CompletionMarker,
    task: PairRunSpec,
    *,
    run_id: str,
    shard_id: str,
) -> None:
    """Reject a completion marker that does not belong to the current task."""
    observed = (marker.run_id, marker.shard_id, marker.protein_id, marker.pair_id, marker.n_samples)
    expected = (run_id, shard_id, task.protein_id, task.pair_id, task.n_samples)
    if observed != expected:
        raise ValueError(
            f"Completion marker identity mismatch: expected {expected}, found {observed}"
        )


def _ensure_remote_completion(
    bundle_dir: Path,
    marker_path: Path,
    marker: CompletionMarker,
    storage: HuggingFaceStorage,
    remote_prefix: str,
) -> None:
    """Re-establish and verify remote durability for a locally completed task."""
    try:
        _verify_remote_completion(marker, storage)
    except Exception:
        storage.upload_bundle(bundle_dir, remote_prefix)
        storage.upload_file(marker_path, marker.remote_marker_path)
        _verify_remote_completion(marker, storage)


def _replace_directory(destination: Path, source: Path) -> None:
    """Replace a directory with rollback if the final move fails."""
    backup: Path | None = None
    try:
        if destination.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{destination.name}.backup-", dir=destination.parent))
            shutil.rmtree(backup)
            destination.replace(backup)
        source.replace(destination)
    except Exception:
        if destination.exists():
            shutil.rmtree(destination)
        if backup is not None and backup.exists():
            backup.replace(destination)
        raise
    finally:
        if backup is not None and backup.exists():
            shutil.rmtree(backup)


def _download_remote_completion_marker(
    storage: HuggingFaceStorage,
    *,
    remote_marker_path: str,
) -> CompletionMarker | None:
    """Return a validated remote completion marker when one already exists."""
    if remote_marker_path not in storage.list_files():
        return None
    with tempfile.TemporaryDirectory(prefix="parallel-bioemu-marker-") as temp_dir:
        marker_path = storage.download_file(remote_marker_path, temp_dir)
        marker = load_valid_completion_marker(marker_path)
    if marker is None:
        raise ValueError(f"Remote completion marker is invalid: {remote_marker_path}")
    return marker


def _verify_remote_completion(
    marker: CompletionMarker,
    storage: HuggingFaceStorage,
) -> None:
    """Assert that the remote manifest and final marker are both visible."""
    storage.verify_remote_files([marker.remote_manifest_path, marker.remote_marker_path])


def _download_remote_chunk_marker(storage: HuggingFaceStorage, remote_marker_path: str) -> ChunkCompletionMarker | None:
    """Download a chunk marker when its immutable remote path exists."""
    if remote_marker_path not in storage.list_files():
        return None
    with tempfile.TemporaryDirectory(prefix="parallel-bioemu-chunk-marker-") as temp_dir:
        marker_path = storage.download_file(remote_marker_path, temp_dir)
        try:
            data = json.loads(marker_path.read_text(encoding="utf-8"))
            marker = ChunkCompletionMarker(**data)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Remote chunk completion marker is invalid: {remote_marker_path}") from exc
    if marker.chunk_n_samples <= 0 or marker.sample_count <= 0:
        raise ValueError(f"Remote chunk completion marker is empty: {remote_marker_path}")
    return marker


def _validate_chunk_marker(marker: ChunkCompletionMarker, chunk: SampleChunkSpec, *, run_id: str, shard_id: str) -> None:
    """Ensure a remote marker belongs to the requested deterministic chunk."""
    expected = (run_id, shard_id, chunk.task.protein_id, chunk.task.pair_id,
                chunk.total_samples, chunk.chunk_index, chunk.n_samples)
    observed = (marker.run_id, marker.shard_id, marker.protein_id, marker.pair_id,
                marker.total_samples, marker.chunk_index, marker.chunk_n_samples)
    if observed != expected:
        raise ValueError(f"Chunk completion marker identity mismatch: expected {expected}, found {observed}")


def _verify_chunk_completion(marker: ChunkCompletionMarker, storage: HuggingFaceStorage) -> None:
    """Verify both the chunk manifest and its final marker are remotely visible."""
    storage.verify_remote_files([marker.remote_manifest_path, marker.remote_marker_path])


def _ensure_local_chunk_bundle(
    marker: ChunkCompletionMarker, remote_prefix: str, bundle_dir: Path, storage: HuggingFaceStorage
) -> ChunkCompletionMarker:
    """Recover a remotely durable chunk onto a fresh runtime before merging."""
    try:
        manifest = load_bioemu_manifest(bundle_dir)
        _validate_sample_files(bundle_dir, manifest)
        return marker
    except (FileNotFoundError, TypeError, ValueError):
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
        storage.download_prefix(remote_prefix, bundle_dir)
        manifest = load_bioemu_manifest(bundle_dir)
        _validate_sample_files(bundle_dir, manifest)
        return replace(marker, bundle_path=str(bundle_dir), sample_count=len(manifest["samples"]))


def _make_heartbeat(
    run_id: str, shard_id: str, *, storage: HuggingFaceStorage, config: WorkerConfig
) -> HeartbeatController | None:
    """Create heartbeat state only when explicitly enabled."""
    if not config.enable_heartbeat:
        return None
    worker_id = config.worker_id or shard_id
    attempt = config.attempt_id if config.attempt_id is not None else 1
    now = datetime.now(timezone.utc).isoformat()
    heartbeat = WorkerHeartbeat(run_id, shard_id, worker_id, attempt, None, None, "starting", now, now)
    local_path = config.workspace_root / run_id / "_heartbeats" / f"{shard_id}_{worker_id}.json"
    # A shard has one active owner. Reusing this path lets the Mac watchdog
    # discover workers from the immutable shard manifest without guessing IDs.
    remote_path = f"{run_id}/_heartbeats/{shard_id}.json"
    return HeartbeatController(heartbeat, local_path, interval_seconds=config.heartbeat_interval_seconds,
                               storage=storage, remote_path=remote_path)
