"""Retry-oriented shard supervision for Colab workers."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .colab_gpu import ColabGpuReport, ensure_allowed_gpu
from .hf_storage import HuggingFaceStorage
from .manifests import ChunkCompletionMarker, CompletionMarker, WorkerConfig
from .worker import run_worker_shard


@dataclass(frozen=True, slots=True)
class SupervisorConfig:
    """Controls shard retry behavior within one currently assigned runtime."""

    workspace_root: Path
    overwrite: bool = False
    max_attempts: int = 1
    retry_delay_seconds: float = 0.0
    allowed_gpu_types: tuple[str, ...] = ("L4", "T4")
    chunk_size: int | None = None
    worker_id: str | None = None
    heartbeat_interval_seconds: float = 120.0
    enable_heartbeat: bool = False

    def __post_init__(self) -> None:
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if self.retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must be non-negative")
        if not self.allowed_gpu_types:
            raise ValueError("allowed_gpu_types must be non-empty")
        if self.chunk_size is not None and self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive when configured")
        if self.heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be positive")


@dataclass(frozen=True, slots=True)
class SupervisorResult:
    """Summary of one supervised shard execution."""

    attempts_used: int
    completions: tuple[CompletionMarker | ChunkCompletionMarker, ...]
    gpu_report: ColabGpuReport


def run_supervised_shard(
    shard_manifest_path: str | Path,
    *,
    storage: HuggingFaceStorage,
    config: SupervisorConfig,
    sleep: Callable[[float], None] = time.sleep,
) -> SupervisorResult:
    """Run one shard with bounded retries on the current runtime.

    This helper can retry transient failures while the runtime is still alive.
    If Colab terminates the VM, an external actor must allocate a new runtime
    and launch the supervisor again.
    """
    gpu_report = ensure_allowed_gpu(config.allowed_gpu_types)
    last_error: Exception | None = None
    for attempt in range(1, config.max_attempts + 1):
        worker_config = WorkerConfig(
            workspace_root=config.workspace_root,
            overwrite=config.overwrite,
            chunk_size=config.chunk_size,
            heartbeat_interval_seconds=config.heartbeat_interval_seconds,
            worker_id=config.worker_id,
            attempt_id=attempt,
            enable_heartbeat=config.enable_heartbeat,
        )
        try:
            completions = run_worker_shard(
                shard_manifest_path,
                storage=storage,
                config=worker_config,
            )
            return SupervisorResult(
                attempts_used=attempt,
                completions=tuple(completions),
                gpu_report=gpu_report,
            )
        except Exception as exc:
            last_error = exc
            if attempt >= config.max_attempts:
                raise
            if config.retry_delay_seconds > 0:
                sleep(config.retry_delay_seconds)
    assert last_error is not None
    raise last_error
