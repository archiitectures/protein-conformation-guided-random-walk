"""Typed manifests and path helpers for explicit BioEmu run orchestration."""

from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _coerce_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _safe_component(value: str, *, field_name: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    banned = {"/", "\\", ".."}
    if any(part in banned for part in text.split("/")) or "/" in text or "\\" in text:
        raise ValueError(f"{field_name} must be a simple path component: {value!r}")
    return text


def build_scoped_output_path(
    root: str | Path,
    *,
    run_id: str,
    protein_id: str,
    pair_id: str,
    n_samples: int,
) -> Path:
    """Return the canonical run/protein/pair/sample-scoped directory."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    base = _coerce_path(root)
    return (
        base
        / _safe_component(run_id, field_name="run_id")
        / _safe_component(protein_id, field_name="protein_id")
        / _safe_component(pair_id, field_name="pair_id")
        / f"n_{n_samples}"
    )


def build_chunk_output_path(
    root: str | Path, *, run_id: str, protein_id: str, pair_id: str,
    total_samples: int, chunk_index: int, chunk_size: int,
) -> Path:
    """Return the immutable local path for one sample chunk."""
    if total_samples <= 0 or chunk_size <= 0 or chunk_index < 0:
        raise ValueError("total_samples and chunk_size must be positive; chunk_index must be non-negative")
    return (build_scoped_output_path(root, run_id=run_id, protein_id=protein_id,
                                     pair_id=pair_id, n_samples=total_samples)
            / "chunks" / f"chunk_{chunk_index:03d}_n_{chunk_size}")


@dataclass(frozen=True, slots=True)
class PairRunSpec:
    """One immutable dataset-pair BioEmu task."""

    protein_id: str
    pair_id: str
    fasta_path: str
    dataset_pair_dir: str
    n_samples: int
    estimated_cost: float
    max_bioemu_samples: int | None = None
    seed: int | None = None
    bioemu_kwargs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        _safe_component(self.protein_id, field_name="protein_id")
        _safe_component(self.pair_id, field_name="pair_id")
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if self.estimated_cost <= 0:
            raise ValueError("estimated_cost must be positive")

    @property
    def storage_key(self) -> str:
        return f"{self.protein_id}/{self.pair_id}/n_{self.n_samples}"


@dataclass(frozen=True, slots=True)
class SampleChunkSpec:
    """A deterministic subset of a :class:`PairRunSpec` sample request."""
    task: PairRunSpec
    chunk_index: int
    sample_start: int
    n_samples: int
    seed: int | None

    def __post_init__(self) -> None:
        if self.chunk_index < 0 or self.sample_start < 0 or self.n_samples <= 0:
            raise ValueError("invalid chunk coordinates")
        if self.sample_start + self.n_samples > self.task.n_samples:
            raise ValueError("chunk exceeds the parent sample request")

    @property
    def total_samples(self) -> int:
        return self.task.n_samples

    @property
    def chunk_id(self) -> str:
        """Stable human- and storage-friendly chunk identifier."""
        return f"chunk_{self.chunk_index:03d}_n_{self.n_samples}"

    @property
    def storage_key(self) -> str:
        return f"{self.task.storage_key}/{self.chunk_id}"


def expand_pair_run_spec(task: PairRunSpec, chunk_size: int = 20) -> tuple[SampleChunkSpec, ...]:
    """Expand a task into stable chunks, deriving independent seeds when needed."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    result = []
    for index, start in enumerate(range(0, task.n_samples, chunk_size)):
        size = min(chunk_size, task.n_samples - start)
        seed = None
        if task.seed is not None:
            digest = hashlib.sha256(f"{task.seed}:{index}".encode("ascii")).digest()
            seed = int.from_bytes(digest[:8], "big", signed=False)
        result.append(SampleChunkSpec(task, index, start, size, seed))
    return tuple(result)


def expand_task_chunks(task: PairRunSpec, chunk_size: int | None = 20) -> tuple[SampleChunkSpec, ...]:
    """Compatibility alias for chunk expansion.

    ``None`` deliberately creates one legacy-shaped chunk rather than disabling
    the returned chunk description entirely.
    """
    if chunk_size is None:
        return (SampleChunkSpec(task, 0, 0, task.n_samples, task.seed),)
    return expand_pair_run_spec(task, chunk_size)


def resume_task_chunks(
    chunks: tuple[SampleChunkSpec, ...] | list[SampleChunkSpec],
    completed_chunk_ids: set[str] | frozenset[str] | list[str],
) -> tuple[SampleChunkSpec, ...]:
    """Return deterministic chunks not represented by completed chunk IDs."""
    completed = set(completed_chunk_ids)
    return tuple(chunk for chunk in chunks if chunk.chunk_id not in completed)


@dataclass(frozen=True, slots=True)
class ShardAssignment:
    """Explicit immutable shard assignment."""

    run_id: str
    shard_id: str
    shard_index: int
    total_shards: int
    tasks: tuple[PairRunSpec, ...]
    total_estimated_cost: float

    def __post_init__(self) -> None:
        _safe_component(self.run_id, field_name="run_id")
        _safe_component(self.shard_id, field_name="shard_id")
        if self.shard_index < 0:
            raise ValueError("shard_index must be non-negative")
        if self.total_shards <= 0:
            raise ValueError("total_shards must be positive")


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Top-level immutable run manifest."""

    run_id: str
    hf_repo_id: str
    pairs: tuple[PairRunSpec, ...]
    shards: tuple[ShardAssignment, ...]

    def __post_init__(self) -> None:
        _safe_component(self.run_id, field_name="run_id")
        if not self.hf_repo_id.strip():
            raise ValueError("hf_repo_id must be non-empty")


@dataclass(frozen=True, slots=True)
class CompletionMarker:
    """Atomic completion metadata written only after validation succeeds."""

    run_id: str
    shard_id: str
    protein_id: str
    pair_id: str
    n_samples: int
    bundle_path: str
    remote_manifest_path: str
    remote_marker_path: str
    sample_count: int
    completed_at_utc: str

    @property
    def storage_key(self) -> str:
        return f"{self.protein_id}/{self.pair_id}/n_{self.n_samples}"


@dataclass(frozen=True, slots=True)
class ChunkCompletionMarker:
    """Durability marker for one independently verified sample chunk."""
    run_id: str
    shard_id: str
    protein_id: str
    pair_id: str
    total_samples: int
    chunk_index: int
    chunk_n_samples: int
    bundle_path: str
    remote_manifest_path: str
    remote_marker_path: str
    sample_count: int
    completed_at_utc: str

    @property
    def storage_key(self) -> str:
        return (f"{self.protein_id}/{self.pair_id}/n_{self.total_samples}/"
                f"chunk_{self.chunk_index:03d}_n_{self.chunk_n_samples}")


@dataclass(frozen=True, slots=True)
class WorkerHeartbeat:
    """Serializable worker liveness state suitable for periodic renewal."""
    run_id: str
    shard_id: str
    worker_id: str
    attempt: str | int
    active_task: str | None
    active_chunk: int | None
    state: str
    started_at_utc: str
    updated_at_utc: str


@dataclass(frozen=True, slots=True)
class RecoveryRecord:
    """A watchdog observation queued for an external supervisor."""
    run_id: str
    shard_id: str
    worker_id: str
    reason: str
    observed_at_utc: str
    heartbeat_path: str
    active_chunk: int | None = None


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    """Worker runtime configuration."""

    workspace_root: Path
    overwrite: bool = False
    # None preserves the original one-bundle worker behavior.
    chunk_size: int | None = None
    heartbeat_interval_seconds: float = 120.0
    worker_id: str | None = None
    attempt_id: str | int | None = None
    enable_heartbeat: bool = False

    def __post_init__(self) -> None:
        if self.chunk_size is not None and self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive when configured")
        if self.heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be positive")


@dataclass(frozen=True, slots=True)
class CollectorConfig:
    """Collector runtime configuration."""

    download_root: Path
    graph_root: Path
    max_processes: int = 1
    overwrite: bool = False

    def __post_init__(self) -> None:
        if self.max_processes <= 0:
            raise ValueError("max_processes must be positive")


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write JSON with a stable trailing newline."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_dataclass_json(path: str | Path, value: Any) -> Path:
    """Serialize a dataclass-like object using ``asdict``."""
    return write_json(path, asdict(value))


def read_json(path: str | Path) -> dict[str, Any]:
    """Load one JSON object."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pair_run_spec_from_dict(data: dict[str, Any]) -> PairRunSpec:
    """Deserialize a pair task."""
    return PairRunSpec(
        protein_id=str(data["protein_id"]),
        pair_id=str(data["pair_id"]),
        fasta_path=str(data["fasta_path"]),
        dataset_pair_dir=str(data["dataset_pair_dir"]),
        n_samples=int(data["n_samples"]),
        estimated_cost=float(data["estimated_cost"]),
        max_bioemu_samples=(
            None if data.get("max_bioemu_samples") is None else int(data["max_bioemu_samples"])
        ),
        seed=None if data.get("seed") is None else int(data["seed"]),
        bioemu_kwargs=dict(data.get("bioemu_kwargs") or {}),
    )


def shard_assignment_from_dict(data: dict[str, Any]) -> ShardAssignment:
    """Deserialize one shard manifest."""
    tasks = tuple(pair_run_spec_from_dict(item) for item in data.get("tasks", []))
    return ShardAssignment(
        run_id=str(data["run_id"]),
        shard_id=str(data["shard_id"]),
        shard_index=int(data["shard_index"]),
        total_shards=int(data["total_shards"]),
        tasks=tasks,
        total_estimated_cost=float(data["total_estimated_cost"]),
    )


def run_manifest_from_dict(data: dict[str, Any]) -> RunManifest:
    """Deserialize the top-level run manifest."""
    pairs = tuple(pair_run_spec_from_dict(item) for item in data.get("pairs", []))
    shards = tuple(shard_assignment_from_dict(item) for item in data.get("shards", []))
    return RunManifest(
        run_id=str(data["run_id"]),
        hf_repo_id=str(data["hf_repo_id"]),
        pairs=pairs,
        shards=shards,
    )
