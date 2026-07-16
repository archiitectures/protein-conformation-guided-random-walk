"""GPU admission helpers for Colab-oriented BioEmu workflows."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Iterable


class ColabGpuError(RuntimeError):
    """Raised when the current runtime GPU does not satisfy the workflow policy."""


@dataclass(frozen=True, slots=True)
class ColabGpuReport:
    """Normalized description of the currently attached NVIDIA GPU."""

    gpu_name: str
    allowed_gpu_types: tuple[str, ...]

    @property
    def normalized_gpu_name(self) -> str:
        return self.gpu_name.upper()

    @property
    def matched_gpu_type(self) -> str | None:
        for gpu_type in self.allowed_gpu_types:
            if gpu_type in self.normalized_gpu_name:
                return gpu_type
        return None

    @property
    def is_allowed(self) -> bool:
        return self.matched_gpu_type is not None

    def describe(self) -> str:
        allowed = ", ".join(self.allowed_gpu_types)
        matched = self.matched_gpu_type or "none"
        return (
            f"Connected GPU: {self.gpu_name} | Allowed GPU types: {allowed} | "
            f"Matched type: {matched}"
        )


def normalize_allowed_gpu_types(allowed_gpu_types: Iterable[str]) -> tuple[str, ...]:
    """Return a de-duplicated tuple of uppercase GPU type tokens."""
    normalized: list[str] = []
    seen: set[str] = set()
    for value in allowed_gpu_types:
        token = value.strip().upper()
        if not token:
            continue
        if token not in seen:
            normalized.append(token)
            seen.add(token)
    if not normalized:
        raise ValueError("allowed_gpu_types must contain at least one non-empty GPU type")
    return tuple(normalized)


def detect_nvidia_gpu_name() -> str:
    """Return the current GPU name from ``nvidia-smi``."""
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True,
    ).strip()


def inspect_current_gpu(allowed_gpu_types: Iterable[str]) -> ColabGpuReport:
    """Inspect the current NVIDIA GPU against the allowed type list."""
    return ColabGpuReport(
        gpu_name=detect_nvidia_gpu_name(),
        allowed_gpu_types=normalize_allowed_gpu_types(allowed_gpu_types),
    )


def ensure_allowed_gpu(allowed_gpu_types: Iterable[str]) -> ColabGpuReport:
    """Raise unless the current GPU name contains one of ``allowed_gpu_types``."""
    report = inspect_current_gpu(allowed_gpu_types)
    if report.is_allowed:
        return report
    allowed = ", ".join(report.allowed_gpu_types)
    raise ColabGpuError(
        f"BioEmu workflow requires one of [{allowed}]; connected GPU is {report.gpu_name!r}."
    )
