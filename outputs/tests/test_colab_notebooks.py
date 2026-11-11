"""Structural tests for the no-Drive VS Code Colab notebook workflow."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_sampling_notebook_is_no_drive_and_contains_every_pair() -> None:
    notebook = _load_notebook("run_bioemu_colab.ipynb")
    source = _all_source(notebook)

    assert notebook["metadata"]["colab"]["gpuType"] == "L4"
    assert "--query-gpu=name" in source
    assert 'ALLOWED_GPU_TYPES = ("L4", "T4")' in source
    assert 'matched_gpu_type = next((gpu for gpu in ALLOWED_GPU_TYPES if gpu in normalized_gpu_name), None)' in source
    assert 'print("Allowed GPU types:", ", ".join(ALLOWED_GPU_TYPES))' in source
    assert "/content/drive" not in source
    assert "drive.mount" not in source
    assert "git clone" not in source.lower()
    assert 'UPLOAD_ARCHIVE = CONTENT_ROOT / f"{EXPERIMENT_LABEL}_workspace.zip"' in source
    assert "create_result_archive" in source
    assert 'run_pair("4AKE_A__1AKE_A", n_samples=2)' in source
    assert "BioEmu still cannot checkpoint mid-sample" in source
    for pair_id in (
        "4AKE_A__1AKE_A",
        "1BPX_A__1BPY_A",
        "1OMP_A__1ANF_A",
        "1CTS_A__2CTS_A",
        "1V4T_A__1V4S_A",
    ):
        assert f'run_pair("{pair_id}", n_samples=200)' in source


def test_colab_workflow_notebook_uses_only_local_imported_paths() -> None:
    notebook = _load_notebook("bioemu_colab_to_graph_workflow.ipynb")
    source = _all_source(notebook)

    assert "/content/drive" not in source
    assert "drive.mount" not in source
    assert 'EXPERIMENT_LABEL = "first_five_proteins"' in source
    assert 'EXPERIMENT_LABEL = "first_five_proteins"  # or "non_ideal_transitions_n3"' in source
    assert "data/bioemu_outputs" in source
    assert "outputs/dataset_bioemu" in source


def test_parallel_worker_notebook_documents_retry_boundary() -> None:
    notebook = _load_notebook("../parallel_bioemu_tools/notebooks/colab_worker.ipynb")
    source = _all_source(notebook)

    assert "ensure_allowed_gpu" in source
    assert "ALLOWED_GPU_TYPES = ('L4', 'T4')" in source
    assert "run_supervised_shard" in source
    assert "cannot restart a dead VM or automate Google account login" in source
    assert "If the marker is already durable in Hugging Face, the rerun will detect it" in source


def test_all_bioemu_notebook_code_cells_parse() -> None:
    for filename in (
        "run_bioemu_colab.ipynb",
        "bioemu_colab_to_graph_workflow.ipynb",
        "analyze_bioemu_outputs.ipynb",
    ):
        notebook = _load_notebook(filename)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                ast.parse(
                    "".join(cell["source"]),
                    filename=f"{filename}:cell-{index}",
                )


def _load_notebook(filename: str) -> dict[str, object]:
    return json.loads((ROOT / "notebooks" / filename).read_text(encoding="utf-8"))


def _all_source(notebook: dict[str, object]) -> str:
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
    )
