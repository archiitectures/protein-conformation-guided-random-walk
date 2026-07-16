"""Sequence-based residue correspondence and structural superposition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from Bio.Align import PairwiseAligner

from protein_conformer_dataset.models import ParsedChain, ResidueMapping


@dataclass(frozen=True, slots=True)
class SequenceMetrics:
    """Summary of a deterministic global sequence alignment."""

    identity: float
    coverage_a: float
    coverage_b: float
    mutation_count: int
    aligned_pairs: tuple[tuple[int, int], ...]


@dataclass(slots=True)
class ChainAlignment:
    """Residue mapping and common C-alpha arrays for two chains."""

    metrics: SequenceMetrics
    residue_mapping: list[ResidueMapping]
    coords_a: np.ndarray
    coords_b: np.ndarray
    aligned_coords_a: np.ndarray
    aligned_coords_b: np.ndarray
    raw_rmsd: float | None
    aligned_rmsd: float | None


def global_alignment(sequence_a: str, sequence_b: str):
    """Return one deterministic global alignment using explicit affine penalties."""
    if not sequence_a or not sequence_b:
        raise ValueError("Both sequences must be nonempty")
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    return aligner.align(sequence_a, sequence_b)[0]


def sequence_metrics(sequence_a: str, sequence_b: str) -> SequenceMetrics:
    """Compute identity and coverage from aligned residue pairs, excluding gaps."""
    alignment = global_alignment(sequence_a, sequence_b)
    pairs: list[tuple[int, int]] = []
    for (start_a, end_a), (start_b, end_b) in zip(alignment.aligned[0], alignment.aligned[1]):
        pairs.extend(zip(range(int(start_a), int(end_a)), range(int(start_b), int(end_b))))
    matches = sum(sequence_a[index_a] == sequence_b[index_b] for index_a, index_b in pairs)
    mutations = len(pairs) - matches
    return SequenceMetrics(
        identity=matches / len(pairs) if pairs else 0.0,
        coverage_a=len(pairs) / len(sequence_a),
        coverage_b=len(pairs) / len(sequence_b),
        mutation_count=mutations,
        aligned_pairs=tuple(pairs),
    )


def map_to_canonical(chain: ParsedChain, canonical_sequence: str) -> None:
    """Annotate structure residues with alignment-derived UniProt positions."""
    metrics = sequence_metrics(chain.sequence, canonical_sequence)
    for structure_index, canonical_index in metrics.aligned_pairs:
        chain.residues[structure_index].uniprot_position = canonical_index + 1


def align_chains(chain_a: ParsedChain, chain_b: ParsedChain) -> ChainAlignment:
    """Align endpoint sequences, extract common C-alpha atoms, and superpose A onto B."""
    metrics = sequence_metrics(chain_a.sequence, chain_b.sequence)
    mapping: list[ResidueMapping] = []
    common_a: list[np.ndarray] = []
    common_b: list[np.ndarray] = []
    for index_a, index_b in metrics.aligned_pairs:
        residue_a = chain_a.residues[index_a]
        residue_b = chain_b.residues[index_b]
        mapping.append(
            ResidueMapping(
                index_a=index_a,
                index_b=index_b,
                auth_seq_id_a=_auth_id(residue_a.auth_seq_id, residue_a.insertion_code),
                auth_seq_id_b=_auth_id(residue_b.auth_seq_id, residue_b.insertion_code),
                residue_name_a=residue_a.residue_name,
                residue_name_b=residue_b.residue_name,
                ca_present_a=residue_a.ca_coord is not None,
                ca_present_b=residue_b.ca_coord is not None,
                uniprot_position_a=residue_a.uniprot_position,
                uniprot_position_b=residue_b.uniprot_position,
                chain_a=chain_a.record.auth_chain_id,
                chain_b=chain_b.record.auth_chain_id,
            )
        )
        if residue_a.ca_coord is not None and residue_b.ca_coord is not None:
            common_a.append(residue_a.ca_coord)
            common_b.append(residue_b.ca_coord)
    coords_a = np.asarray(common_a, dtype=float).reshape((-1, 3))
    coords_b = np.asarray(common_b, dtype=float).reshape((-1, 3))
    if len(coords_a) == 0:
        return ChainAlignment(metrics, mapping, coords_a, coords_b, coords_a.copy(), coords_b.copy(), None, None)
    raw_rmsd = float(np.sqrt(np.mean(np.sum((coords_a - coords_b) ** 2, axis=1))))
    aligned_a, aligned_b = kabsch_superpose(coords_a, coords_b)
    aligned_rmsd = float(np.sqrt(np.mean(np.sum((aligned_a - aligned_b) ** 2, axis=1))))
    return ChainAlignment(
        metrics=metrics,
        residue_mapping=mapping,
        coords_a=coords_a,
        coords_b=coords_b,
        aligned_coords_a=aligned_a,
        aligned_coords_b=aligned_b,
        raw_rmsd=raw_rmsd,
        aligned_rmsd=aligned_rmsd,
    )


def kabsch_superpose(coords_a: np.ndarray, coords_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigidly superpose A onto B and return both arrays in B's coordinate frame."""
    a = np.asarray(coords_a, dtype=float)
    b = np.asarray(coords_b, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"Coordinate arrays must have equal shape (L, 3); got {a.shape} and {b.shape}")
    if len(a) == 0 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Coordinate arrays must be nonempty and finite")
    centroid_a = a.mean(axis=0)
    centroid_b = b.mean(axis=0)
    centered_a = a - centroid_a
    centered_b = b - centroid_b
    covariance = centered_a.T @ centered_b
    u_matrix, _, vt_matrix = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(u_matrix @ vt_matrix))
    rotation = u_matrix @ correction @ vt_matrix
    return centered_a @ rotation + centroid_b, b.copy()


def _auth_id(number: str | int | None, insertion_code: str) -> str | int | None:
    if number is None:
        return None
    return f"{number}{insertion_code}" if insertion_code else number
