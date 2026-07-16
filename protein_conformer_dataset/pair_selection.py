"""Endpoint comparison, duplicate handling, and candidate ranking."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from protein_conformer_dataset.alignment import align_chains
from protein_conformer_dataset.config import QCConfig, RankingConfig
from protein_conformer_dataset.models import EndpointPair, PairStatus, ParsedChain, StructureRecord
from protein_conformer_dataset.quality_control import evaluate_pair_quality


def prefilter_structure_records(
    records: list[StructureRecord],
    canonical_sequence_length: int | None,
    maximum: int,
) -> list[StructureRecord]:
    """Rank metadata before downloads and retain a configurable bounded set."""
    def key(record: StructureRecord) -> tuple[float, float, str]:
        if canonical_sequence_length and record.sequence_length:
            coverage_error = abs(record.sequence_length - canonical_sequence_length) / canonical_sequence_length
        else:
            coverage_error = 1.0
        resolution = record.resolution if record.resolution is not None else 99.0
        return (coverage_error, resolution, record.structure_id)

    return sorted(records, key=key)[:maximum]


def compare_pair(
    protein_id: str,
    chain_a: ParsedChain,
    chain_b: ParsedChain,
    qc: QCConfig,
    *,
    state_a: str | None = None,
    state_b: str | None = None,
) -> EndpointPair:
    """Align and QC one endpoint pair without interpreting it as a transition."""
    alignment = align_chains(chain_a, chain_b)
    quality = evaluate_pair_quality(chain_a, chain_b, alignment, qc)
    return EndpointPair(
        protein_id=protein_id,
        uniprot_id=chain_a.record.uniprot_id or chain_b.record.uniprot_id,
        structure_a=chain_a.record,
        structure_b=chain_b.record,
        quality=quality,
        residue_mapping=alignment.residue_mapping,
        coords_a=alignment.coords_a,
        coords_b=alignment.coords_b,
        aligned_coords_a=alignment.aligned_coords_a,
        aligned_coords_b=alignment.aligned_coords_b,
        state_a=state_a,
        state_b=state_b,
    )


def compare_all_pairs(
    protein_id: str,
    chains: list[ParsedChain],
    qc: QCConfig,
) -> list[EndpointPair]:
    """Compute pairwise evidence for a bounded set of parsed chains."""
    return [compare_pair(protein_id, a, b, qc) for a, b in combinations(chains, 2)]


def rank_candidate_pairs(
    pairs: list[EndpointPair],
    ranking: RankingConfig,
    duplicate_rmsd_angstrom: float,
) -> list[EndpointPair]:
    """Rank structurally distinct endpoints with a nonphysical selection score."""
    usable = [
        pair
        for pair in pairs
        if pair.quality.status is not PairStatus.REJECTED
        and pair.quality.ca_rmsd is not None
        and pair.quality.ca_rmsd >= duplicate_rmsd_angstrom
    ]
    if not usable:
        return []
    rmsd_values = np.asarray([pair.quality.ca_rmsd for pair in usable], dtype=float)
    minimum = float(rmsd_values.min())
    span = float(np.ptp(rmsd_values))
    for pair in usable:
        quality = pair.quality
        normalized_rmsd = 1.0 if span == 0 else (float(quality.ca_rmsd) - minimum) / span
        coverage = min(quality.aligned_coverage_a, quality.aligned_coverage_b)
        missing = (quality.missing_fraction_a + quality.missing_fraction_b) / 2.0
        resolutions = [
            value
            for value in (pair.structure_a.resolution, pair.structure_b.resolution)
            if value is not None
        ]
        resolution_quality = 0.0 if not resolutions else 1.0 / (1.0 + float(np.mean(resolutions)))
        pair.candidate_score = (
            ranking.rmsd_weight * normalized_rmsd
            + ranking.coverage_weight * coverage
            + ranking.identity_weight * quality.sequence_identity
            + ranking.resolution_weight * resolution_quality
            - ranking.mutation_penalty * quality.mutation_count
            - ranking.missing_penalty * missing
        )
    status_priority = {PairStatus.ACCEPTED: 1, PairStatus.MANUAL_REVIEW: 0}
    return sorted(
        usable,
        key=lambda pair: (
            status_priority[pair.quality.status],
            pair.candidate_score,
            pair.pair_id,
        ),
        reverse=True,
    )
