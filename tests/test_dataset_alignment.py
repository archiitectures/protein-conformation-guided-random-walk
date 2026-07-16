"""Alignment and superposition tests for experimental endpoint preparation."""

from __future__ import annotations

import numpy as np

from protein_conformer_dataset.alignment import align_chains, kabsch_superpose, sequence_metrics
from protein_conformer_dataset.models import ParsedChain, ParsedResidue, StructureRecord


def test_global_sequence_alignment_reports_identity_coverage_and_mutation() -> None:
    metrics = sequence_metrics("ACDEFG", "ACXDEFG")

    assert metrics.identity == 1.0
    assert metrics.coverage_a == 1.0
    assert metrics.coverage_b == 6 / 7
    assert metrics.mutation_count == 0


def test_sequence_alignment_detects_substitution() -> None:
    metrics = sequence_metrics("ACDEFG", "ACNEFG")

    assert metrics.identity == 5 / 6
    assert metrics.mutation_count == 1


def test_residue_mapping_retains_missing_ca_positions() -> None:
    chain_a = _chain("A", [True, True, False, True])
    chain_b = _chain("B", [True, False, True, True])

    result = align_chains(chain_a, chain_b)

    assert len(result.residue_mapping) == 4
    assert len(result.coords_a) == 2
    assert result.residue_mapping[2].ca_present_a is False
    assert result.residue_mapping[1].ca_present_b is False


def test_kabsch_returns_zero_for_rotated_translated_coordinates() -> None:
    coords = np.asarray([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=float)
    rotation = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    transformed = coords @ rotation + np.asarray([8.0, -2.0, 3.0])

    aligned, target = kabsch_superpose(coords, transformed)

    assert np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))) < 1e-10


def _chain(chain_id: str, ca_mask: list[bool]) -> ParsedChain:
    residues = []
    for index, present in enumerate(ca_mask):
        residues.append(
            ParsedResidue(
                index=index,
                residue_name="ALA",
                one_letter="A",
                auth_seq_id=index + 1,
                label_seq_id=index + 1,
                ca_coord=np.asarray([index * 3.8, 0.0, 0.0]) if present else None,
            )
        )
    return ParsedChain(StructureRecord("TEST", "1", chain_id), residues)
