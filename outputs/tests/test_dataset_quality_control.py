"""QC status and warning tests."""

from __future__ import annotations

import numpy as np

from protein_conformer_dataset.alignment import align_chains
from protein_conformer_dataset.config import QCConfig
from protein_conformer_dataset.models import PairStatus, ParsedChain, ParsedResidue, StructureRecord
from protein_conformer_dataset.quality_control import evaluate_pair_quality


def test_low_coverage_pair_is_rejected() -> None:
    chain_a = _chain("A" * 50, "P1")
    chain_b = _chain("A" * 20, "P1")
    config = QCConfig(minimum_common_ca_count=10, minimum_aligned_coverage=0.85)

    report = evaluate_pair_quality(chain_a, chain_b, align_chains(chain_a, chain_b), config)

    assert report.status is PairStatus.REJECTED
    assert any("coverage" in reason.lower() for reason in report.rejection_reasons)


def test_environment_difference_requires_manual_review() -> None:
    chain_a = _chain("A" * 45, "P1")
    chain_b = _chain("A" * 45, "P1")
    chain_b.record.ligand_ids = ["ATP"]

    report = evaluate_pair_quality(chain_a, chain_b, align_chains(chain_a, chain_b), QCConfig())

    assert report.status is PairStatus.MANUAL_REVIEW
    assert any("Ligand" in warning for warning in report.warnings)


def test_strict_mode_rejects_mutations() -> None:
    chain_a = _chain("A" * 45, "P1")
    chain_b = _chain("A" * 44 + "G", "P1")

    report = evaluate_pair_quality(
        chain_a, chain_b, align_chains(chain_a, chain_b), QCConfig.from_mode("strict")
    )

    assert report.status is PairStatus.REJECTED
    assert any("mutations" in reason.lower() for reason in report.rejection_reasons)


def _chain(sequence: str, uniprot_id: str) -> ParsedChain:
    record = StructureRecord("TEST", "1", "A", uniprot_id=uniprot_id, organism="Synthetic")
    residues = [
        ParsedResidue(
            index=index,
            residue_name="ALA" if letter == "A" else "GLY",
            one_letter=letter,
            auth_seq_id=index + 1,
            label_seq_id=index + 1,
            ca_coord=np.asarray([index * 3.8, 0.0, 0.0]),
        )
        for index, letter in enumerate(sequence)
    ]
    return ParsedChain(record, residues)
