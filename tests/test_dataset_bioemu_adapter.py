"""Tests for dataset-pair adaptation of Colab BioEmu output bundles."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from guided_conformational_path.bioemu_io import write_bioemu_bundle_manifest
from guided_conformational_path.dataset_bioemu_adapter import (
    DatasetBioEmuAdapterError,
    load_adapted_bioemu_nodes,
    load_dataset_pair_selection,
)
from guided_conformational_path.dataset_bioemu_experiment import (
    DatasetBioEmuConfig,
    run_dataset_bioemu_experiment,
)
from guided_conformational_path.io import write_pdb_ca
from guided_conformational_path.large_synthetic import write_run_outputs
from scripts.run_dataset_bioemu_experiment import parse_args


def test_adapter_subsets_full_sequence_samples_to_common_residues(tmp_path: Path) -> None:
    pair_dir = _write_fake_pair(tmp_path)
    bundle_dir = _write_fake_bundle(tmp_path, sequence="AAAAAA", n_samples=2)
    selection = load_dataset_pair_selection(pair_dir)

    nodes = load_adapted_bioemu_nodes(selection, bundle_dir)

    assert selection.residue_positions.tolist() == [1, 3, 6]
    assert selection.endpoint_a.n_residues == 3
    assert len(nodes) == 2
    assert nodes[0].coords.shape == (3, 3)
    assert nodes[0].metadata["full_sequence_ca_count"] == 6
    assert nodes[0].metadata["selected_ca_count"] == 3
    assert nodes[0].metadata["residue_positions"] == [1, 3, 6]
    assert nodes[0].metadata["sequence_verified"] is True
    np.testing.assert_allclose(nodes[0].coords, selection.endpoint_a.coords, atol=1e-3)


def test_adapter_rejects_bundle_sequence_mismatch(tmp_path: Path) -> None:
    pair_dir = _write_fake_pair(tmp_path)
    bundle_dir = _write_fake_bundle(tmp_path, sequence="AAAATA", n_samples=1)
    selection = load_dataset_pair_selection(pair_dir)

    with pytest.raises(DatasetBioEmuAdapterError, match="does not match"):
        load_adapted_bioemu_nodes(selection, bundle_dir)


def test_adapter_rejects_mapping_position_outside_selected_sequence(tmp_path: Path) -> None:
    pair_dir = _write_fake_pair(tmp_path, positions=(1, 3, 7))

    with pytest.raises(DatasetBioEmuAdapterError, match="exceeds selected sequence length"):
        load_dataset_pair_selection(pair_dir)


def test_dataset_bioemu_experiment_runs_graph_and_writes_outputs(tmp_path: Path) -> None:
    pair_dir = _write_fake_pair(tmp_path)
    bundle_dir = _write_fake_bundle(tmp_path, sequence="AAAAAA", n_samples=3)
    config = DatasetBioEmuConfig(
        dataset_pair_dir=pair_dir,
        bioemu_bundle_dir=bundle_dir,
        outputs_dir=tmp_path / "outputs",
        write_pymol=False,
        write_py3dmol=False,
        write_animation=False,
    )

    run = run_dataset_bioemu_experiment(config)
    write_run_outputs(run, config)

    summary = json.loads((config.outputs_dir / "experiment_summary.json").read_text())
    assert run.path_result.node_ids[0] == "A"
    assert run.path_result.node_ids[-1] == "B"
    assert run.graph.number_of_nodes() == 5
    assert summary["input"]["mode"] == "dataset_bioemu"
    assert summary["input"]["common_ca_count"] == 3
    assert summary["input"]["sequence_verified"] is True
    assert summary["n_generated_nodes"] == 3
    assert (config.outputs_dir / "graph.png").exists()
    assert (config.outputs_dir / "path_summary.json").exists()


def test_dataset_bioemu_cli_parses_required_handoff_paths(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dataset_bioemu_experiment.py",
            "--pair-dir",
            "pair",
            "--bioemu-bundle-dir",
            "bundle",
            "--max-bioemu-samples",
            "7",
            "--no-animation",
            "--no-pymol",
            "--no-py3dmol",
        ],
    )

    args = parse_args()

    assert args.pair_dir == Path("pair")
    assert args.bioemu_bundle_dir == Path("bundle")
    assert args.max_bioemu_samples == 7
    assert args.write_animation is False
    assert args.write_pymol is False
    assert args.write_py3dmol is False


def _write_fake_pair(
    tmp_path: Path,
    positions: tuple[int, int, int] = (1, 3, 6),
) -> Path:
    pair_dir = tmp_path / "dataset" / "PTEST" / "pairs" / "A__B"
    processed_dir = pair_dir / "processed"
    processed_dir.mkdir(parents=True)
    sequence_dir = pair_dir / "sequence"
    sequence_dir.mkdir()
    (sequence_dir / "bioemu_selected.fasta").write_text(
        ">PTEST_canonical\nAAAAAA\n", encoding="utf-8"
    )
    full_coords = _full_coords(6)
    indices = np.clip(np.asarray(positions) - 1, 0, len(full_coords) - 1)
    endpoint_a = full_coords[indices]
    endpoint_b = endpoint_a.copy()
    endpoint_b[1, 1] += 0.8
    np.save(processed_dir / "endpoint_A_aligned_ca.npy", endpoint_a)
    np.save(processed_dir / "endpoint_B_aligned_ca.npy", endpoint_b)

    fieldnames = [
        "index_a",
        "index_b",
        "auth_seq_id_a",
        "auth_seq_id_b",
        "residue_name_a",
        "residue_name_b",
        "ca_present_a",
        "ca_present_b",
        "uniprot_position_a",
        "uniprot_position_b",
        "chain_a",
        "chain_b",
    ]
    with (processed_dir / "common_residue_mapping.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, position in enumerate(positions):
            writer.writerow(
                {
                    "index_a": index,
                    "index_b": index,
                    "auth_seq_id_a": position,
                    "auth_seq_id_b": position,
                    "residue_name_a": "ALA",
                    "residue_name_b": "ALA",
                    "ca_present_a": True,
                    "ca_present_b": True,
                    "uniprot_position_a": position,
                    "uniprot_position_b": position,
                    "chain_a": "A",
                    "chain_b": "A",
                }
            )

    manifest = {
        "protein_id": "PTEST",
        "uniprot_id": "PTEST",
        "pair_id": "A__B",
        "sequence_selection": {
            "canonical_sequence": "AAAAAA",
            "construct_sequence": None,
            "selected_sequence": "AAAAAA",
            "source": "canonical",
            "warnings": [],
        },
        "paths": {
            "endpoint_a_aligned_ca": "processed/endpoint_A_aligned_ca.npy",
            "endpoint_b_aligned_ca": "processed/endpoint_B_aligned_ca.npy",
            "bioemu_sequence": "sequence/bioemu_selected.fasta",
        },
    }
    (pair_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return pair_dir


def _write_fake_bundle(tmp_path: Path, sequence: str, n_samples: int) -> Path:
    bundle_dir = tmp_path / f"bundle_{sequence}"
    nodes_dir = bundle_dir / "nodes"
    nodes_dir.mkdir(parents=True)
    samples: list[dict[str, object]] = []
    for sample_index in range(n_samples):
        coords = _full_coords(len(sequence))
        if sample_index:
            coords[:, 1] += 0.1 * sample_index * np.sin(np.arange(len(sequence)))
        pdb_path = nodes_dir / f"bioemu_{sample_index:03d}.pdb"
        write_pdb_ca(pdb_path, coords)
        samples.append(
            {
                "node_id": f"bioemu_{sample_index:03d}",
                "pdb_path": str(pdb_path.relative_to(bundle_dir)),
            }
        )
    write_bioemu_bundle_manifest(
        bundle_dir,
        samples=samples,
        sequence_id="PTEST",
        sequence=sequence,
        n_samples_requested=n_samples,
    )
    return bundle_dir


def _full_coords(length: int) -> np.ndarray:
    index = np.arange(length, dtype=float)
    return np.column_stack([3.8 * index, 0.2 * np.sin(index), 0.1 * np.cos(index)])
