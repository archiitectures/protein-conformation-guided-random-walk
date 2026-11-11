from __future__ import annotations

import pytest

from parallel_bioemu_tools.colab_gpu import (
    ColabGpuError,
    ensure_allowed_gpu,
    inspect_current_gpu,
    normalize_allowed_gpu_types,
)


def test_normalize_allowed_gpu_types_deduplicates_and_uppercases() -> None:
    assert normalize_allowed_gpu_types([" l4 ", "T4", "l4"]) == ("L4", "T4")


def test_inspect_current_gpu_reports_match(monkeypatch) -> None:
    monkeypatch.setattr(
        "parallel_bioemu_tools.colab_gpu.detect_nvidia_gpu_name",
        lambda: "NVIDIA Tesla T4",
    )
    report = inspect_current_gpu(["L4", "T4"])
    assert report.is_allowed is True
    assert report.matched_gpu_type == "T4"
    assert "Allowed GPU types: L4, T4" in report.describe()


def test_ensure_allowed_gpu_rejects_unexpected_runtime(monkeypatch) -> None:
    monkeypatch.setattr(
        "parallel_bioemu_tools.colab_gpu.detect_nvidia_gpu_name",
        lambda: "NVIDIA A100-SXM4-40GB",
    )
    with pytest.raises(ColabGpuError, match="requires one of \\[L4, T4\\]"):
        ensure_allowed_gpu(["L4", "T4"])
