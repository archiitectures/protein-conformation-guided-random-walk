"""Reusable execution unit for one dataset-pair BioEmu Colab job."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from guided_conformational_path.bioemu_io import BIOEMU_BUNDLE_SCHEMA
from guided_conformational_path.bioemu_runner import read_fasta, run_bioemu_from_fasta
from guided_conformational_path.colab_transfer import create_result_archive, sha256_file
from guided_conformational_path.dataset_bioemu_experiment import (
    DatasetBioEmuConfig,
    run_dataset_bioemu_experiment,
)
from guided_conformational_path.large_synthetic import format_run_summary, write_run_outputs


def run_colab_pair(
    pair_id: str,
    n_samples: int,
    *,
    workspace_dir: str | Path = "/content/protein-conformation-guided-random-walk",
    content_root: str | Path = "/content",
    graph_k_neighbors: int = 10,
    write_visual_outputs: bool = True,
    path_top_k: int = 4,
    min_path_edge_diversity: float = 0.20,
    path_candidate_scan_limit: int = 1000,
    num_top_gifs_generated: int = 6,
    num_bottom_gifs_generated: int = 1,
    write_alternative_path_visuals: bool = False,
    seed: int = 7,
) -> dict[str, Any]:
    """Sample, graph, and archive one exported endpoint pair on Colab.

    A completed BioEmu bundle is reused when its manifest already exists. This
    gives task-level resume after a controller interruption, although BioEmu
    itself still cannot checkpoint partway through one sampling call.

    Args:
        pair_id: Exported dataset pair identifier, such as
            ``"1CTS_A__2CTS_A"``.
        n_samples: Number of BioEmu conformations to request.
        workspace_dir: Extracted repository root on the remote runtime.
        content_root: Colab filesystem root used for runs and result exports.
        graph_k_neighbors: Number of nearest neighbors used to sparsify the
            conformation graph. Zero selects a fully connected graph.
        write_visual_outputs: Whether to write PyMOL, py3Dmol, and GIF outputs.
        seed: BioEmu sampling seed.

    Returns:
        JSON-compatible result metadata including the result archive path and
        SHA-256 digest.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    workspace = Path(workspace_dir).expanduser().resolve()
    content = Path(content_root).expanduser().resolve()
    dataset_root = workspace / "data/dataset/first_five_proteins"
    pair_manifests = {
        path.parent.name: path
        for path in sorted(dataset_root.glob("*/pairs/*/manifest.json"))
    }
    if pair_id not in pair_manifests:
        available = ", ".join(sorted(pair_manifests))
        raise KeyError(f"Unknown pair_id {pair_id!r}; available pairs: {available}")

    pair_manifest_path = pair_manifests[pair_id]
    pair_manifest = json.loads(pair_manifest_path.read_text(encoding="utf-8"))
    pair_dir = pair_manifest_path.parent
    protein_id = str(pair_manifest["protein_id"])
    fasta_path = pair_dir / pair_manifest["paths"]["bioemu_sequence"]
    run_name = f"n_{n_samples}"
    bundle_dir = content / "runs/bioemu_bundles" / protein_id / pair_id / run_name
    graph_dir = content / "runs/graph_outputs" / protein_id / pair_id / run_name
    archive_path = content / "exports" / f"{protein_id}__{pair_id}__{run_name}.zip"

    print(f"\n[{protein_id} / {pair_id}] requesting {n_samples} samples", flush=True)
    if _is_reusable_bundle(bundle_dir, fasta_path, n_samples):
        print(f"Reusing completed BioEmu bundle: {bundle_dir}", flush=True)
    else:
        if bundle_dir.exists():
            print("Discarding incomplete or mismatched BioEmu bundle.", flush=True)
            shutil.rmtree(bundle_dir)
        run_bioemu_from_fasta(
            fasta_path=fasta_path,
            output_dir=bundle_dir,
            n_samples=n_samples,
            seed=seed,
        )

    config = DatasetBioEmuConfig(
        dataset_pair_dir=pair_dir,
        bioemu_bundle_dir=bundle_dir,
        max_bioemu_samples=n_samples,
        k_neighbors=graph_k_neighbors,
        outputs_dir=graph_dir,
        write_nodes=False,
        write_pymol=write_visual_outputs,
        write_py3dmol=write_visual_outputs,
        write_animation=write_visual_outputs,
        path_top_k=path_top_k,
        min_path_edge_diversity=min_path_edge_diversity,
        path_candidate_scan_limit=path_candidate_scan_limit,
        num_top_gifs_generated=num_top_gifs_generated,
        num_bottom_gifs_generated=num_bottom_gifs_generated,
        write_candidate_report=True,
        write_alternative_path_visuals=write_alternative_path_visuals,
    )
    run = run_dataset_bioemu_experiment(config)
    write_run_outputs(run, config)
    create_result_archive(bundle_dir, graph_dir, archive_path)

    result: dict[str, Any] = {
        "protein_id": protein_id,
        "pair_id": pair_id,
        "requested_samples": n_samples,
        "retained_samples": run.summary["n_generated_nodes"],
        "path": run.path_result.node_ids,
        "total_cost": run.path_result.total_cost,
        "archive": str(archive_path),
        "archive_sha256": sha256_file(archive_path),
    }
    print(format_run_summary(run, graph_dir), flush=True)
    print(f"Completed result archive: {archive_path}", flush=True)
    return result


def _is_reusable_bundle(bundle_dir: Path, fasta_path: Path, n_samples: int) -> bool:
    """Return whether an existing bundle is complete for the requested task."""
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _, expected_sequence = read_fasta(fasta_path)
        sequence = manifest.get("sequence", {})
        samples = manifest.get("samples")
        if manifest.get("schema_version") != BIOEMU_BUNDLE_SCHEMA:
            return False
        if int(manifest.get("n_samples_requested", 0)) != n_samples:
            return False
        if sequence.get("sequence") != expected_sequence:
            return False
        if not isinstance(samples, list) or not samples:
            return False
        for sample in samples:
            if not isinstance(sample, dict) or not sample.get("pdb_path"):
                return False
            sample_path = bundle_dir / str(sample["pdb_path"])
            if not sample_path.is_file() or sample_path.stat().st_size == 0:
                return False
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return False
    return True
