"""Portable BioEmu bundle input/output helpers.

BioEmu itself is expected to run in a Linux/CUDA environment such as Google
Colab. The local macOS workflow consumes a portable output bundle containing a
``manifest.json`` file and normalized PDB files with C-alpha coordinates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from guided_conformational_path.io import read_pdb_ca
from guided_conformational_path.nodes import ConformationNode


BIOEMU_BUNDLE_SCHEMA = "bioemu_bundle_v1"


def read_fasta(path: str | Path) -> tuple[str, str]:
    """Read a single FASTA record and return ``(record_id, sequence)``."""
    fasta_path = Path(path)
    record_id = fasta_path.stem
    sequence_parts: list[str] = []
    with fasta_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(">"):
                if sequence_parts:
                    break
                record_id = stripped[1:].split()[0] or record_id
                continue
            sequence_parts.append(stripped)
    sequence = "".join(sequence_parts).replace(" ", "").upper()
    if not sequence:
        raise ValueError(f"No amino acid sequence found in FASTA: {fasta_path}")
    return record_id, sequence


def write_bioemu_bundle_manifest(
    bundle_dir: str | Path,
    samples: list[dict[str, Any]],
    sequence_id: str,
    sequence: str,
    n_samples_requested: int,
    endpoint_a_path: str | Path | None = None,
    endpoint_b_path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a BioEmu bundle manifest and return its path.

    Sample dictionaries must include ``pdb_path``. Relative paths are interpreted
    relative to ``bundle_dir`` when the bundle is loaded.
    """
    bundle_path = Path(bundle_dir)
    bundle_path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": BIOEMU_BUNDLE_SCHEMA,
        "sequence": {
            "id": sequence_id,
            "sequence": sequence,
            "length": len(sequence),
        },
        "n_samples_requested": n_samples_requested,
        "inputs": {
            "endpoint_a_path": str(endpoint_a_path) if endpoint_a_path is not None else None,
            "endpoint_b_path": str(endpoint_b_path) if endpoint_b_path is not None else None,
        },
        "samples": samples,
        "metadata": metadata or {},
    }
    manifest_path = bundle_path / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest_path


def load_bioemu_manifest(bundle_dir: str | Path) -> dict[str, Any]:
    """Load and minimally validate a BioEmu bundle manifest."""
    bundle_path = Path(bundle_dir)
    manifest_path = bundle_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"BioEmu bundle manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    schema = manifest.get("schema_version")
    if schema != BIOEMU_BUNDLE_SCHEMA:
        raise ValueError(
            f"Unsupported BioEmu bundle schema {schema!r}; expected {BIOEMU_BUNDLE_SCHEMA!r}"
        )
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise ValueError("BioEmu bundle manifest must contain a 'samples' list")
    return manifest


def load_bioemu_bundle(
    bundle_dir: str | Path,
    expected_n_residues: int | None = None,
    max_samples: int | None = None,
) -> list[ConformationNode]:
    """Load normalized BioEmu PDB samples as neutral external nodes.

    Args:
        bundle_dir: Directory containing ``manifest.json``.
        expected_n_residues: Optional required number of C-alpha coordinates per
            sample. Use this to enforce compatibility with endpoint A/B.
        max_samples: Optional cap for local graph construction.
    """
    bundle_path = Path(bundle_dir)
    manifest = load_bioemu_manifest(bundle_path)
    sequence = manifest.get("sequence", {})
    samples = manifest["samples"]
    if max_samples is not None:
        if max_samples < 0:
            raise ValueError("max_samples must be non-negative")
        samples = samples[:max_samples]

    nodes: list[ConformationNode] = []
    for sample_idx, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"BioEmu sample entry {sample_idx} must be an object")
        pdb_path_value = sample.get("pdb_path")
        if not pdb_path_value:
            raise ValueError(f"BioEmu sample entry {sample_idx} is missing 'pdb_path'")
        pdb_path = _resolve_bundle_path(bundle_path, pdb_path_value)
        if not pdb_path.exists():
            raise FileNotFoundError(f"BioEmu sample PDB not found: {pdb_path}")
        coords = read_pdb_ca(pdb_path)
        if expected_n_residues is not None and coords.shape[0] != expected_n_residues:
            raise ValueError(
                "BioEmu sample residue count mismatch for "
                f"{pdb_path}: expected {expected_n_residues}, got {coords.shape[0]}"
            )
        node_id = str(sample.get("node_id") or f"bioemu_{sample_idx:03d}")
        nodes.append(
            ConformationNode(
                node_id=node_id,
                source="external",
                coords=coords,
                metadata={
                    "generator": "BioEmu",
                    "sample_index": sample_idx,
                    "sequence_id": sequence.get("id"),
                    "sequence_length": sequence.get("length"),
                    "manifest_path": str(bundle_path / "manifest.json"),
                    "pdb_path": str(pdb_path),
                    "raw_path": sample.get("raw_path"),
                    "bioemu_metadata": sample.get("metadata", {}),
                },
                file_path=pdb_path,
            )
        )
    if not nodes:
        raise ValueError(f"BioEmu bundle contains no samples: {bundle_path}")
    return nodes


def _resolve_bundle_path(bundle_dir: Path, value: str | Path) -> Path:
    """Resolve absolute or bundle-relative paths."""
    path = Path(value)
    if path.is_absolute():
        return path
    return bundle_dir / path
