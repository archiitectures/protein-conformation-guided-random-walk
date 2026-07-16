"""Parallel BioEmu orchestration helpers."""

from .colab_gpu import ColabGpuError, ColabGpuReport, ensure_allowed_gpu, inspect_current_gpu
from .collector import collect_run
from .manifests import (
    ChunkCompletionMarker,
    CollectorConfig,
    CompletionMarker,
    PairRunSpec,
    RecoveryRecord,
    RunManifest,
    SampleChunkSpec,
    ShardAssignment,
    WorkerHeartbeat,
    WorkerConfig,
    build_chunk_output_path,
    build_scoped_output_path,
    expand_pair_run_spec,
    expand_task_chunks,
    resume_task_chunks,
)
from .heartbeat import HeartbeatController, WorkerHeartbeatController
from .merge import assemble_durable_chunks, assemble_merged_bundle, merge_chunk_bundles, merge_chunks
from .supervisor import SupervisorConfig, SupervisorResult, run_supervised_shard
from .watchdog import MacWatchdog
from .worker import (
    load_valid_chunk_completion_marker,
    run_chunked_task,
    run_one_task,
    run_worker_shard,
    write_chunk_completion_marker,
)

__all__ = [
    "ColabGpuError",
    "ColabGpuReport",
    "ChunkCompletionMarker",
    "CollectorConfig",
    "CompletionMarker",
    "PairRunSpec",
    "RecoveryRecord",
    "RunManifest",
    "SampleChunkSpec",
    "ShardAssignment",
    "SupervisorConfig",
    "SupervisorResult",
    "WorkerConfig",
    "WorkerHeartbeat",
    "HeartbeatController",
    "WorkerHeartbeatController",
    "MacWatchdog",
    "assemble_durable_chunks",
    "assemble_merged_bundle",
    "merge_chunk_bundles",
    "merge_chunks",
    "build_chunk_output_path",
    "build_scoped_output_path",
    "collect_run",
    "ensure_allowed_gpu",
    "inspect_current_gpu",
    "expand_pair_run_spec",
    "expand_task_chunks",
    "resume_task_chunks",
    "run_chunked_task",
    "run_one_task",
    "load_valid_chunk_completion_marker",
    "write_chunk_completion_marker",
    "run_supervised_shard",
    "run_worker_shard",
]
