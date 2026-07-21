"""Alignment and superposition tests for experimental endpoint preparation."""

from __future__ import annotations

import numpy as np

from protein_conformer_dataset.alignment import (
    align_chains,
    kabsch_superpose,
    select_shared_domain_from_b,
    select_shared_exact_matches,
    sequence_metrics,
)
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


def test_shared_domain_selection_trims_full_length_endpoint_to_reference_construct() -> None:
    full = _sequence_chain("A", "MSTACDEFGHIK")
    domain = _sequence_chain("B", "ACDEFG")

    selected_a, selected_b, metadata = select_shared_domain_from_b(full, domain)

    assert selected_a.sequence == "ACDEFG"
    assert selected_b.sequence == "ACDEFG"
    assert metadata["mode"] == "shared_domain_from_b"
    assert metadata["selected_residue_count"] == 6
    assert metadata["endpoint_a_original_indices"] == [3, 4, 5, 6, 7, 8]


def test_shared_domain_selection_rejects_incompatible_reference() -> None:
    full = _sequence_chain("A", "MSTACDEFGHIK")
    domain = _sequence_chain("B", "VVVVVV")

    with np.testing.assert_raises_regex(ValueError, "high sequence identity"):
        select_shared_domain_from_b(full, domain)


def test_shared_domain_selection_excludes_unmatched_reference_expression_tag() -> None:
    full = _sequence_chain("A", "MSTACDEFGHIK")
    tagged_domain = _sequence_chain("B", "GACDEFGHIK")

    selected_a, selected_b, metadata = select_shared_domain_from_b(full, tagged_domain)

    assert selected_a.sequence == selected_b.sequence == "ACDEFGHIK"
    assert metadata["reference_coverage"] == 0.9


def test_exact_match_selection_retains_only_mutation_invariant_residues() -> None:
    chain_a = _sequence_chain("A", "ACDEFGHIKLMN")
    chain_b = _sequence_chain("B", "ACDEYGHIKLMQ")

    selected_a, selected_b, metadata = select_shared_exact_matches(
        chain_a,
        chain_b,
        minimum_identity=0.80,
        minimum_selected_residues=5,
    )

    assert selected_a.sequence == selected_b.sequence == "ACDEGHIKLM"
    assert metadata["excluded_nonidentical_residue_count"] == 2


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


def _sequence_chain(chain_id: str, sequence: str) -> ParsedChain:
    residues = [
        ParsedResidue(
            index=index,
            residue_name="ALA",
            one_letter=letter,
            auth_seq_id=index + 1,
            label_seq_id=index + 1,
            ca_coord=np.asarray([index * 3.8, 0.0, 0.0]),
        )
        for index, letter in enumerate(sequence)
    ]
    return ParsedChain(StructureRecord("TEST", "1", chain_id), residues)
