"""Adapt full-sequence BioEmu samples to an exported endpoint dataset pair.

The dataset generator stores endpoint coordinates only for residues represented
by C-alpha atoms in both structures. BioEmu, by contrast, generates coordinates
for the complete selected input sequence. This module validates that both sides
refer to the same sequence, selects the shared endpoint residues from every
BioEmu sample, and places the selected coordinates in the endpoint coordinate
frame for graph analysis and visualization.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from guided_conformational_path.bioemu_io import (
    load_bioemu_bundle,
    load_bioemu_manifest,
    read_fasta,
)
from guided_conformational_path.nodes import ConformationNode


class DatasetBioEmuAdapterError(ValueError):
    """Raised when a dataset pair and BioEmu bundle cannot be mapped safely."""


@dataclass(slots=True)
class DatasetPairSelection:
    """Validated dataset-pair inputs needed to adapt BioEmu samples.

    Attributes:
        pair_dir: Root directory of one exported endpoint pair.
        manifest_path: Path to the pair's ``manifest.json``.
        manifest: Parsed pair manifest retained for provenance.
        endpoint_a: Aligned endpoint A node in endpoint B's coordinate frame.
        endpoint_b: Aligned endpoint B node in its original coordinate frame.
        selected_sequence: Exact sequence supplied to BioEmu.
        residue_positions: One-based positions selected from each full sample.
        mapping_source: ``canonical`` for UniProt positions or ``construct`` for
            pairwise construct indices.
        protein_id: Dataset protein identifier, normally a UniProt accession.
        protein_name: Preferred local protein name when present in the manifest.
        uniprot_id: UniProt accession when present in the manifest.
        pair_id: Stable endpoint-pair identifier.
    """

    pair_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    endpoint_a: ConformationNode
    endpoint_b: ConformationNode
    selected_sequence: str
    residue_positions: np.ndarray
    mapping_source: str
    protein_id: str
    protein_name: str | None
    uniprot_id: str | None
    pair_id: str


def load_dataset_pair_selection(pair_dir_or_manifest: str | Path) -> DatasetPairSelection:
    """Load an exported dataset pair and derive its BioEmu residue selection.

    ``pair_dir_or_manifest`` may point to the pair directory or directly to its
    manifest. The aligned endpoint arrays are used because the dataset export
    already superposes endpoint A onto endpoint B. Mapping is accepted only when
    every selected coordinate has a unique, ordered position in the exact
    sequence selected for BioEmu.
    """
    supplied_path = Path(pair_dir_or_manifest).expanduser()
    manifest_path = supplied_path if supplied_path.is_file() else supplied_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset pair manifest not found: {manifest_path}")
    pair_dir = manifest_path.parent
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    sequence_selection = manifest.get("sequence_selection")
    if not isinstance(sequence_selection, dict):
        raise DatasetBioEmuAdapterError("Dataset pair manifest is missing 'sequence_selection'")
    selected_sequence = _normalized_sequence(sequence_selection.get("selected_sequence"))
    source = str(sequence_selection.get("source") or "").lower()
    if source not in {"canonical", "construct"}:
        raise DatasetBioEmuAdapterError(
            "BioEmu sequence source must be 'canonical' or 'construct'; "
            f"got {source!r}"
        )

    paths = manifest.get("paths")
    if not isinstance(paths, dict):
        raise DatasetBioEmuAdapterError("Dataset pair manifest is missing 'paths'")
    _validate_exported_fasta(pair_dir, paths, selected_sequence)
    endpoint_a_path = _required_pair_path(pair_dir, paths, "endpoint_a_aligned_ca")
    endpoint_b_path = _required_pair_path(pair_dir, paths, "endpoint_b_aligned_ca")
    coords_a = _load_endpoint_array(endpoint_a_path, "endpoint A")
    coords_b = _load_endpoint_array(endpoint_b_path, "endpoint B")
    if coords_a.shape != coords_b.shape:
        raise DatasetBioEmuAdapterError(
            f"Aligned endpoint shapes differ: {coords_a.shape} and {coords_b.shape}"
        )

    mapping_rows = _load_mapping_rows(pair_dir, manifest)
    residue_positions = _derive_residue_positions(mapping_rows, source)
    if len(residue_positions) != len(coords_a):
        raise DatasetBioEmuAdapterError(
            "Common residue mapping and aligned endpoint arrays disagree: "
            f"mapping selects {len(residue_positions)} residues but endpoints contain {len(coords_a)}"
        )
    if len(residue_positions) == 0:
        raise DatasetBioEmuAdapterError("Dataset pair has no common C-alpha residues")
    if residue_positions[-1] > len(selected_sequence):
        raise DatasetBioEmuAdapterError(
            f"Residue position {residue_positions[-1]} exceeds selected sequence length "
            f"{len(selected_sequence)}"
        )

    protein_id = str(manifest.get("protein_id") or manifest.get("uniprot_id") or pair_dir.parent.name)
    protein_name = _preferred_protein_name(manifest)
    uniprot_id = _preferred_uniprot_id(manifest)
    pair_id = str(manifest.get("pair_id") or pair_dir.name)
    shared_metadata = {
        "dataset_pair_manifest": str(manifest_path),
        "protein_id": protein_id,
        "pair_id": pair_id,
        "selected_sequence_length": len(selected_sequence),
        "residue_positions": residue_positions.tolist(),
        "mapping_source": source,
        "coordinate_frame": "dataset_endpoint_b",
    }
    endpoint_a = ConformationNode(
        "A",
        "A_endpoint",
        coords_a,
        metadata={**shared_metadata, "dataset_endpoint": "A"},
        file_path=endpoint_a_path,
    )
    endpoint_b = ConformationNode(
        "B",
        "B_endpoint",
        coords_b,
        metadata={**shared_metadata, "dataset_endpoint": "B"},
        file_path=endpoint_b_path,
    )
    return DatasetPairSelection(
        pair_dir=pair_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        endpoint_a=endpoint_a,
        endpoint_b=endpoint_b,
        selected_sequence=selected_sequence,
        residue_positions=residue_positions,
        mapping_source=source,
        protein_id=protein_id,
        protein_name=protein_name,
        uniprot_id=uniprot_id,
        pair_id=pair_id,
    )


def load_adapted_bioemu_nodes(
    selection: DatasetPairSelection,
    bundle_dir: str | Path,
    max_samples: int | None = 200,
) -> list[ConformationNode]:
    """Load, validate, subset, and align BioEmu samples for one dataset pair.

    The bundle sequence must exactly match the dataset pair's selected BioEmu
    sequence. Each sample must contain one C-alpha coordinate per sequence
    residue. Rigid alignment changes only the global pose, not the sample's
    internal conformation or Kabsch-RMSD graph costs.
    """
    bundle_path = Path(bundle_dir).expanduser()
    manifest = load_bioemu_manifest(bundle_path)
    sequence_metadata = manifest.get("sequence")
    bundle_sequence = _normalized_sequence(
        sequence_metadata.get("sequence") if isinstance(sequence_metadata, dict) else None
    )
    declared_length = (
        _as_optional_int(sequence_metadata.get("length"))
        if isinstance(sequence_metadata, dict)
        else None
    )
    if declared_length is not None and declared_length != len(bundle_sequence):
        raise DatasetBioEmuAdapterError(
            "BioEmu bundle sequence metadata is inconsistent: "
            f"declared length {declared_length}, actual length {len(bundle_sequence)}"
        )
    if bundle_sequence != selection.selected_sequence:
        raise DatasetBioEmuAdapterError(
            "BioEmu bundle sequence does not match the dataset pair selected sequence "
            f"(bundle length {len(bundle_sequence)}, dataset length {len(selection.selected_sequence)})"
        )

    full_nodes = load_bioemu_bundle(
        bundle_path,
        expected_n_residues=len(selection.selected_sequence),
        max_samples=max_samples,
    )
    indices = selection.residue_positions - 1
    adapted: list[ConformationNode] = []
    used_ids = {"A", "B"}
    for sample_index, node in enumerate(full_nodes):
        selected_coords = node.coords[indices]
        aligned_coords = _superpose_onto(selected_coords, selection.endpoint_a.coords)
        node_id = _unique_node_id(node.node_id, sample_index, used_ids)
        used_ids.add(node_id)
        adapted.append(
            ConformationNode(
                node_id=node_id,
                source="external",
                coords=aligned_coords,
                metadata={
                    **node.metadata,
                    "adapter": "dataset_bioemu",
                    "dataset_pair_manifest": str(selection.manifest_path),
                    "protein_id": selection.protein_id,
                    "pair_id": selection.pair_id,
                    "full_sequence_ca_count": node.n_residues,
                    "selected_ca_count": len(selection.residue_positions),
                    "residue_positions": selection.residue_positions.tolist(),
                    "mapping_source": selection.mapping_source,
                    "sequence_verified": True,
                    "coordinate_frame": "dataset_endpoint_b",
                },
                file_path=node.file_path,
            )
        )
    return adapted


def _load_mapping_rows(pair_dir: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Load mapping rows from the standardized CSV, with manifest fallback."""
    mapping_path = pair_dir / "processed" / "common_residue_mapping.csv"
    if mapping_path.exists():
        with mapping_path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    rows = manifest.get("residue_mapping")
    if not isinstance(rows, list):
        raise DatasetBioEmuAdapterError(
            f"Residue mapping not found at {mapping_path} or in the pair manifest"
        )
    return [row for row in rows if isinstance(row, dict)]


def _derive_residue_positions(rows: list[dict[str, Any]], source: str) -> np.ndarray:
    """Return ordered one-based sequence positions for common endpoint C-alphas."""
    positions: list[int] = []
    for row_index, row in enumerate(rows):
        if not (_as_bool(row.get("ca_present_a")) and _as_bool(row.get("ca_present_b"))):
            continue
        if source == "canonical":
            position_a = _as_optional_int(row.get("uniprot_position_a"))
            position_b = _as_optional_int(row.get("uniprot_position_b"))
            if position_a is None or position_b is None or position_a != position_b:
                raise DatasetBioEmuAdapterError(
                    f"Common mapping row {row_index} lacks one shared UniProt position"
                )
            position = position_a
        else:
            index_a = _as_optional_int(row.get("index_a"))
            index_b = _as_optional_int(row.get("index_b"))
            if index_a is None or index_b is None or index_a != index_b:
                raise DatasetBioEmuAdapterError(
                    f"Common mapping row {row_index} lacks one shared construct index"
                )
            position = index_a + 1
        positions.append(position)

    array = np.asarray(positions, dtype=int)
    if np.any(array < 1) or (len(array) > 1 and np.any(np.diff(array) <= 0)):
        raise DatasetBioEmuAdapterError(
            "Selected residue positions must be unique, positive, and strictly increasing"
        )
    return array


def _superpose_onto(coords: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Rigidly superpose coordinates onto an equal-shaped reference array."""
    moving = np.asarray(coords, dtype=float)
    target = np.asarray(reference, dtype=float)
    if moving.shape != target.shape or moving.ndim != 2 or moving.shape[1] != 3:
        raise DatasetBioEmuAdapterError(
            f"Cannot align coordinate shapes {moving.shape} and {target.shape}"
        )
    centered_moving = moving - moving.mean(axis=0)
    centered_target = target - target.mean(axis=0)
    covariance = centered_moving.T @ centered_target
    u_matrix, _, vt_matrix = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(u_matrix @ vt_matrix))
    rotation = u_matrix @ correction @ vt_matrix
    return centered_moving @ rotation + target.mean(axis=0)


def _required_pair_path(pair_dir: Path, paths: dict[str, Any], key: str) -> Path:
    """Resolve one required pair-manifest path and verify it exists."""
    value = paths.get(key)
    if not value:
        raise DatasetBioEmuAdapterError(f"Dataset pair manifest path {key!r} is missing")
    path = Path(value)
    resolved = path if path.is_absolute() else pair_dir / path
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset pair file not found: {resolved}")
    return resolved


def _validate_exported_fasta(
    pair_dir: Path,
    paths: dict[str, Any],
    selected_sequence: str,
) -> None:
    """Verify the exported BioEmu FASTA when the pair manifest references it."""
    value = paths.get("bioemu_sequence")
    if not value:
        return
    fasta_path = Path(value)
    resolved = fasta_path if fasta_path.is_absolute() else pair_dir / fasta_path
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset pair BioEmu FASTA not found: {resolved}")
    _, fasta_sequence = read_fasta(resolved)
    if _normalized_sequence(fasta_sequence) != selected_sequence:
        raise DatasetBioEmuAdapterError(
            "Dataset pair BioEmu FASTA does not match manifest sequence_selection"
        )


def _load_endpoint_array(path: Path, label: str) -> np.ndarray:
    """Load and validate one aligned endpoint coordinate array."""
    coords = np.asarray(np.load(path, allow_pickle=False), dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3 or not np.all(np.isfinite(coords)):
        raise DatasetBioEmuAdapterError(
            f"{label} coordinates must be finite with shape (L, 3); got {coords.shape}"
        )
    return coords


def _normalized_sequence(value: Any) -> str:
    """Normalize an amino-acid sequence and reject absent or malformed values."""
    sequence = "".join(str(value or "").split()).upper()
    if not sequence:
        raise DatasetBioEmuAdapterError("A nonempty BioEmu sequence is required")
    if not sequence.isalpha():
        raise DatasetBioEmuAdapterError("BioEmu sequence must contain alphabetic residue codes only")
    return sequence


def _as_bool(value: Any) -> bool:
    """Parse booleans emitted by JSON or CSV exports."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _as_optional_int(value: Any) -> int | None:
    """Parse an optional integer field from JSON or CSV."""
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise DatasetBioEmuAdapterError(f"Expected an integer mapping value, got {value!r}") from exc


def _unique_node_id(candidate: str, sample_index: int, used_ids: set[str]) -> str:
    """Return a graph-safe BioEmu node ID without endpoint collisions."""
    base = str(candidate or f"bioemu_{sample_index:03d}")
    if base not in used_ids:
        return base
    prefixed = f"bioemu_{base}"
    suffix = 1
    unique = prefixed
    while unique in used_ids:
        suffix += 1
        unique = f"{prefixed}_{suffix}"
    return unique


def _preferred_protein_name(manifest: dict[str, Any]) -> str | None:
    for candidate in (
        manifest.get("protein_name"),
        manifest.get("structure_a", {}).get("protein_name") if isinstance(manifest.get("structure_a"), dict) else None,
        manifest.get("structure_b", {}).get("protein_name") if isinstance(manifest.get("structure_b"), dict) else None,
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _preferred_uniprot_id(manifest: dict[str, Any]) -> str | None:
    for candidate in (
        manifest.get("uniprot_id"),
        manifest.get("structure_a", {}).get("uniprot_id") if isinstance(manifest.get("structure_a"), dict) else None,
        manifest.get("structure_b", {}).get("uniprot_id") if isinstance(manifest.get("structure_b"), dict) else None,
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None
