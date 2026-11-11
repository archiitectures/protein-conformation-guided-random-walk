"""Tests for BioEmu bundle import and local precomputed analysis."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from guided_conformational_path.bioemu_experiment import (
    BioEmuPrecomputedConfig,
    run_bioemu_precomputed_experiment,
)
from guided_conformational_path.bioemu_io import (
    load_bioemu_bundle,
    read_fasta,
    write_bioemu_bundle_manifest,
)
from guided_conformational_path.bioemu_runner import (
    _resolve_bioemu_sampler,
    _safe_batch_size_100,
    run_bioemu_from_fasta,
)
from guided_conformational_path.io import write_pdb_ca
from guided_conformational_path.large_synthetic import write_run_outputs
from scripts.run_bioemu_precomputed_experiment import parse_args


def test_read_fasta_returns_single_sequence(tmp_path: Path) -> None:
    fasta_path = tmp_path / "sequence.fasta"
    fasta_path.write_text(">toy\nACD\nEFG\n", encoding="utf-8")

    record_id, sequence = read_fasta(fasta_path)

    assert record_id == "toy"
    assert sequence == "ACDEFG"


def test_read_fasta_uses_only_first_record(tmp_path: Path) -> None:
    fasta_path = tmp_path / "multi_sequence.fasta"
    fasta_path.write_text(">first\nACD\n>second\nWXY\n", encoding="utf-8")

    record_id, sequence = read_fasta(fasta_path)

    assert record_id == "first"
    assert sequence == "ACD"


def test_load_bioemu_bundle_returns_external_nodes(tmp_path: Path) -> None:
    bundle_dir = _write_fake_bioemu_bundle(tmp_path, n_residues=6, n_samples=3)

    nodes = load_bioemu_bundle(bundle_dir, expected_n_residues=6)

    assert len(nodes) == 3
    assert {node.source for node in nodes} == {"external"}
    assert nodes[0].node_id == "bioemu_000"
    assert nodes[0].metadata["generator"] == "BioEmu"
    assert nodes[0].metadata["sequence_id"] == "toy"
    assert "manifest_path" in nodes[0].metadata


def test_load_bioemu_bundle_rejects_missing_pdb(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    write_bioemu_bundle_manifest(
        bundle_dir,
        samples=[{"node_id": "missing", "pdb_path": "nodes/missing.pdb"}],
        sequence_id="toy",
        sequence="ACDEFG",
        n_samples_requested=1,
    )

    try:
        load_bioemu_bundle(bundle_dir)
    except FileNotFoundError as exc:
        assert "missing.pdb" in str(exc)
    else:
        raise AssertionError("Expected missing sample PDB to raise FileNotFoundError")


def test_load_bioemu_bundle_rejects_residue_count_mismatch(tmp_path: Path) -> None:
    bundle_dir = _write_fake_bioemu_bundle(tmp_path, n_residues=5, n_samples=1)

    try:
        load_bioemu_bundle(bundle_dir, expected_n_residues=6)
    except ValueError as exc:
        assert "residue count mismatch" in str(exc)
    else:
        raise AssertionError("Expected residue-count mismatch to raise ValueError")


def test_importing_bioemu_io_does_not_import_bioemu() -> None:
    assert "bioemu" not in sys.modules


def test_bioemu_runner_fake_sampler_writes_loadable_bundle(tmp_path: Path) -> None:
    fasta_path = tmp_path / "sequence.fasta"
    fasta_path.write_text(">toy\nAAAAAA\n", encoding="utf-8")

    def fake_sampler(sequence: str, n_samples: int, **_: object) -> np.ndarray:
        assert sequence == "AAAAAA"
        assert n_samples == 2
        return np.stack([_coords(6, 0.1), _coords(6, 0.2)], axis=0)

    manifest = run_bioemu_from_fasta(
        fasta_path,
        tmp_path / "runner_bundle",
        n_samples=2,
        seed=1,
        sampler=fake_sampler,
    )
    nodes = load_bioemu_bundle(tmp_path / "runner_bundle", expected_n_residues=6)

    assert manifest["schema_version"] == "bioemu_bundle_v1"
    assert manifest["n_samples_requested"] == 2
    assert len(manifest["samples"]) == 2
    assert len(nodes) == 2
    assert nodes[0].metadata["generator"] == "BioEmu"


def test_bioemu_runner_resolves_official_main_entry_point() -> None:
    def main() -> None:
        return None

    assert _resolve_bioemu_sampler([SimpleNamespace(main=main)]) is main


def test_bioemu_runner_prevents_zero_batch_for_long_sequence(tmp_path: Path) -> None:
    fasta_path = tmp_path / "sequence.fasta"
    fasta_path.write_text(f">long\n{'A' * 350}\n", encoding="utf-8")
    captured: dict[str, int] = {}

    def fake_sampler(
        sequence: str,
        n_samples: int,
        batch_size_100: int = 10,
    ) -> np.ndarray:
        captured["batch_size_100"] = batch_size_100
        return np.stack([_coords(len(sequence), 0.1)] * n_samples)

    run_bioemu_from_fasta(
        fasta_path,
        tmp_path / "long_bundle",
        n_samples=1,
        sampler=fake_sampler,
    )

    assert captured["batch_size_100"] == 13
    assert int(captured["batch_size_100"] * (100 / 350) ** 2) >= 1
    assert _safe_batch_size_100(214) == 10


def test_bioemu_runner_expands_standard_xtc_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Official topology/XTC output becomes one PDB node per frame."""
    fasta_path = tmp_path / "sequence.fasta"
    fasta_path.write_text(">toy\nAAAA\n", encoding="utf-8")

    class FakeTopology:
        def select(self, query: str) -> np.ndarray:
            assert query == "name CA"
            return np.array([1, 4, 7, 10])

    xyz = np.zeros((3, 12, 3), dtype=float)
    for frame_index in range(3):
        xyz[frame_index, [1, 4, 7, 10], 0] = np.arange(4) * 0.38
        xyz[frame_index, [1, 4, 7, 10], 1] = frame_index * 0.01
    fake_mdtraj = SimpleNamespace(
        load=lambda xtc, top: SimpleNamespace(xyz=xyz, topology=FakeTopology())
    )
    monkeypatch.setitem(sys.modules, "mdtraj", fake_mdtraj)

    def fake_sampler(output_dir: str, **_: object) -> None:
        raw_dir = Path(output_dir)
        (raw_dir / "topology.pdb").write_text("END\n", encoding="utf-8")
        (raw_dir / "samples.xtc").write_bytes(b"fake")

    manifest = run_bioemu_from_fasta(
        fasta_path,
        tmp_path / "xtc_bundle",
        n_samples=3,
        sampler=fake_sampler,
    )
    nodes = load_bioemu_bundle(tmp_path / "xtc_bundle", expected_n_residues=4)

    assert len(nodes) == 3
    assert len(manifest["samples"]) == 3
    assert nodes[0].metadata["bioemu_metadata"]["source_format"] == "xtc_trajectory"
    np.testing.assert_allclose(nodes[0].coords[:, 0], np.arange(4) * 3.8, atol=1e-3)


def test_run_bioemu_precomputed_experiment_writes_outputs(tmp_path: Path) -> None:
    endpoint_a, endpoint_b = _write_fake_endpoints(tmp_path, n_residues=8)
    bundle_dir = _write_fake_bioemu_bundle(tmp_path, n_residues=8, n_samples=4)
    config = BioEmuPrecomputedConfig(
        endpoint_a_path=endpoint_a,
        endpoint_b_path=endpoint_b,
        bioemu_bundle_dir=bundle_dir,
        outputs_dir=tmp_path / "outputs",
        write_pymol=False,
        write_py3dmol=False,
        write_animation=False,
    )

    run = run_bioemu_precomputed_experiment(config)
    write_run_outputs(run, config)

    summary = json.loads((config.outputs_dir / "experiment_summary.json").read_text(encoding="utf-8"))
    assert run.path_result.node_ids[0] == "A"
    assert run.path_result.node_ids[-1] == "B"
    assert summary["input"]["mode"] == "bioemu_precomputed"
    assert summary["input"]["bioemu_bundle_dir"] == str(bundle_dir)
    assert summary["n_generated_nodes"] == 4
    assert (config.outputs_dir / "graph.png").exists()
    assert (config.outputs_dir / "path_summary.json").exists()


def test_bioemu_precomputed_cli_parses_flags(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_bioemu_precomputed_experiment.py",
            "--endpoint-a",
            "A.pdb",
            "--endpoint-b",
            "B.pdb",
            "--bioemu-bundle-dir",
            "bundle",
            "--max-bioemu-samples",
            "7",
            "--no-pymol",
            "--no-py3dmol",
            "--no-animation",
        ],
    )

    args = parse_args()

    assert args.endpoint_a == Path("A.pdb")
    assert args.endpoint_b == Path("B.pdb")
    assert args.bioemu_bundle_dir == Path("bundle")
    assert args.max_bioemu_samples == 7
    assert args.write_pymol is False
    assert args.write_py3dmol is False
    assert args.write_animation is False


def _write_fake_endpoints(tmp_path: Path, n_residues: int) -> tuple[Path, Path]:
    coords_a = _coords(n_residues, offset=0.0)
    coords_b = _coords(n_residues, offset=2.0)
    endpoint_a = tmp_path / "A.pdb"
    endpoint_b = tmp_path / "B.pdb"
    write_pdb_ca(endpoint_a, coords_a)
    write_pdb_ca(endpoint_b, coords_b)
    return endpoint_a, endpoint_b


def _write_fake_bioemu_bundle(tmp_path: Path, n_residues: int, n_samples: int) -> Path:
    bundle_dir = tmp_path / "bioemu_bundle"
    nodes_dir = bundle_dir / "nodes"
    nodes_dir.mkdir(parents=True)
    samples = []
    for sample_idx in range(n_samples):
        pdb_path = nodes_dir / f"bioemu_{sample_idx:03d}.pdb"
        write_pdb_ca(pdb_path, _coords(n_residues, offset=0.3 * (sample_idx + 1)))
        samples.append(
            {
                "node_id": f"bioemu_{sample_idx:03d}",
                "pdb_path": str(pdb_path.relative_to(bundle_dir)),
                "raw_path": f"raw/sample_{sample_idx:03d}.pdb",
                "metadata": {"fake_score": float(sample_idx)},
            }
        )
    write_bioemu_bundle_manifest(
        bundle_dir,
        samples=samples,
        sequence_id="toy",
        sequence="A" * n_residues,
        n_samples_requested=n_samples,
    )
    return bundle_dir


def _coords(n_residues: int, offset: float) -> np.ndarray:
    x = np.arange(n_residues, dtype=float) * 3.8
    return np.column_stack([x, np.full(n_residues, offset), np.zeros(n_residues)])
