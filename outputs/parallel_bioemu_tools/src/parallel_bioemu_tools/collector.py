"""Local collector for validated remote BioEmu bundles."""

from __future__ import annotations

import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from guided_conformational_path.bioemu_io import load_bioemu_manifest
from guided_conformational_path.dataset_bioemu_adapter import (
    load_adapted_bioemu_nodes,
    load_dataset_pair_selection,
)
from guided_conformational_path.dataset_bioemu_experiment import (
    DatasetBioEmuConfig,
    run_dataset_bioemu_experiment,
)
from guided_conformational_path.large_synthetic import write_run_outputs

from .hf_storage import HuggingFaceStorage
from .manifests import (
    CollectorConfig,
    PairRunSpec,
    build_scoped_output_path,
    read_json,
    run_manifest_from_dict,
)
from .worker import load_valid_completion_marker


def collect_run(
    run_manifest_path: str | Path,
    *,
    storage: HuggingFaceStorage,
    config: CollectorConfig,
) -> list[dict[str, Any]]:
    """Download validated bundles and run local graph collection."""
    run_manifest = run_manifest_from_dict(read_json(run_manifest_path))
    tasks = list(run_manifest.pairs)
    if config.max_processes == 1:
        return [
            _collect_one_task(task, run_id=run_manifest.run_id, storage=storage, config=config)
            for task in tasks
        ]

    with ProcessPoolExecutor(max_workers=config.max_processes) as pool:
        futures = [
            pool.submit(
                _collect_one_task_process,
                asdict(task),
                run_manifest.run_id,
                run_manifest.hf_repo_id,
                str(config.download_root),
                str(config.graph_root),
                config.overwrite,
            )
            for task in tasks
        ]
        return [future.result() for future in futures]


def _collect_one_task_process(
    task_data: dict[str, Any],
    run_id: str,
    repo_id: str,
    download_root: str,
    graph_root: str,
    overwrite: bool,
) -> dict[str, Any]:
    """Process entry point for concurrent collection."""
    storage = HuggingFaceStorage(repo_id=repo_id)
    config = CollectorConfig(
        download_root=Path(download_root),
        graph_root=Path(graph_root),
        max_processes=1,
        overwrite=overwrite,
    )
    task = PairRunSpec(
        protein_id=str(task_data["protein_id"]),
        pair_id=str(task_data["pair_id"]),
        fasta_path=str(task_data["fasta_path"]),
        dataset_pair_dir=str(task_data["dataset_pair_dir"]),
        n_samples=int(task_data["n_samples"]),
        estimated_cost=float(task_data["estimated_cost"]),
        max_bioemu_samples=task_data.get("max_bioemu_samples"),
        seed=task_data.get("seed"),
        bioemu_kwargs=dict(task_data.get("bioemu_kwargs") or {}),
    )
    return _collect_one_task(task, run_id=run_id, storage=storage, config=config)


def _collect_one_task(
    task: PairRunSpec,
    *,
    run_id: str,
    storage: HuggingFaceStorage,
    config: CollectorConfig,
) -> dict[str, Any]:
    """Download one bundle and run the existing graph pipeline."""
    bundle_dir = build_scoped_output_path(
        config.download_root,
        run_id=run_id,
        protein_id=task.protein_id,
        pair_id=task.pair_id,
        n_samples=task.n_samples,
    )
    graph_dir = build_scoped_output_path(
        config.graph_root,
        run_id=run_id,
        protein_id=task.protein_id,
        pair_id=task.pair_id,
        n_samples=task.n_samples,
    )
    if config.overwrite:
        for existing_dir in (bundle_dir, graph_dir):
            if existing_dir.exists():
                shutil.rmtree(existing_dir)
    else:
        if bundle_dir.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing download directory: {bundle_dir}"
            )
        if graph_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing graph directory: {graph_dir}")

    remote_prefix = f"{run_id}/{task.protein_id}/{task.pair_id}/n_{task.n_samples}"
    remote_marker_path = f"{remote_prefix}/_worker_completed.json"
    storage.verify_remote_files([remote_marker_path])
    local_bundle_dir = storage.download_prefix(remote_prefix, bundle_dir)
    marker = load_valid_completion_marker(local_bundle_dir / "_worker_completed.json")
    if marker is None:
        raise ValueError(f"Downloaded bundle has no valid completion marker: {local_bundle_dir}")
    marker_identity = (marker.run_id, marker.protein_id, marker.pair_id, marker.n_samples)
    expected_identity = (run_id, task.protein_id, task.pair_id, task.n_samples)
    if marker_identity != expected_identity:
        raise ValueError(
            f"Downloaded completion marker identity mismatch: expected {expected_identity}, "
            f"found {marker_identity}"
        )
    manifest = load_bioemu_manifest(local_bundle_dir)
    expected_manifest_path = f"{remote_prefix}/manifest.json"
    if marker.remote_manifest_path != expected_manifest_path:
        raise ValueError(
            f"Completion marker points to {marker.remote_manifest_path!r}, "
            f"expected {expected_manifest_path!r}"
        )
    if marker.remote_marker_path != remote_marker_path:
        raise ValueError("Completion marker remote path does not match its bundle prefix")
    if len(manifest.get("samples", [])) != marker.sample_count:
        raise ValueError("Completion marker sample count does not match the bundle manifest")
    if len(manifest.get("samples", [])) == 0:
        raise ValueError(f"Downloaded bundle is empty: {local_bundle_dir}")

    selection = load_dataset_pair_selection(Path(task.dataset_pair_dir))
    loaded_nodes = load_adapted_bioemu_nodes(
        selection,
        local_bundle_dir,
        max_samples=task.max_bioemu_samples,
    )
    if not loaded_nodes:
        raise ValueError(f"Validated bundle yielded no adapted nodes: {local_bundle_dir}")

    experiment_config = DatasetBioEmuConfig(
        dataset_pair_dir=Path(task.dataset_pair_dir),
        bioemu_bundle_dir=local_bundle_dir,
        max_bioemu_samples=task.max_bioemu_samples,
        outputs_dir=graph_dir,
    )
    run = run_dataset_bioemu_experiment(experiment_config)
    graph_dir.mkdir(parents=True, exist_ok=True)
    write_run_outputs(run, experiment_config)
    return {
        "protein_id": task.protein_id,
        "pair_id": task.pair_id,
        "bundle_dir": str(local_bundle_dir),
        "graph_dir": str(graph_dir),
        "sample_count": len(manifest["samples"]),
    }
