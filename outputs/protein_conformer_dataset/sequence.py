"""Amino-acid normalization and endpoint sequence selection."""

from __future__ import annotations

from dataclasses import dataclass


STANDARD_AMINO_ACIDS = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O",
}

COMMON_MODIFICATIONS = {
    "MSE": "M",
    "SEP": "S",
    "TPO": "T",
    "PTR": "Y",
    "HYP": "P",
    "CSO": "C",
    "KCX": "K",
    "CME": "C",
}


def residue_to_one_letter(residue_name: str) -> str | None:
    """Map a standard or well-understood modified amino acid to one letter."""
    normalized = residue_name.strip().upper()
    return STANDARD_AMINO_ACIDS.get(normalized) or COMMON_MODIFICATIONS.get(normalized)


@dataclass(frozen=True, slots=True)
class SequenceChoice:
    """Sequence selected for downstream BioEmu use and its rationale."""

    canonical_sequence: str | None
    construct_sequence: str | None
    selected_sequence: str | None
    source: str | None
    warnings: list[str]


def choose_bioemu_sequence(
    sequence_a: str,
    sequence_b: str,
    canonical_sequence: str | None,
    *,
    minimum_identity: float = 0.98,
    minimum_coverage: float = 0.85,
) -> SequenceChoice:
    """Choose a sequence consistent with both endpoint constructs.

    BioEmu should not be given a full canonical sequence when both experimental
    endpoints are a shared truncation, nor an ambiguous construct when endpoint
    sequences materially disagree.
    """
    from protein_conformer_dataset.alignment import sequence_metrics

    warnings: list[str] = []
    endpoint_metrics = sequence_metrics(sequence_a, sequence_b)
    if endpoint_metrics.identity < minimum_identity:
        warnings.append("Endpoint constructs do not have sufficient sequence identity for BioEmu")
        return SequenceChoice(canonical_sequence, None, None, None, warnings)

    common_construct = sequence_a if sequence_a == sequence_b else None
    if canonical_sequence:
        metrics_a = sequence_metrics(sequence_a, canonical_sequence)
        metrics_b = sequence_metrics(sequence_b, canonical_sequence)
        canonical_compatible = (
            metrics_a.identity >= minimum_identity
            and metrics_b.identity >= minimum_identity
            and metrics_a.coverage_b >= minimum_coverage
            and metrics_b.coverage_b >= minimum_coverage
            and metrics_a.coverage_a >= minimum_coverage
            and metrics_b.coverage_a >= minimum_coverage
        )
        if canonical_compatible:
            return SequenceChoice(canonical_sequence, common_construct, canonical_sequence, "canonical", warnings)

    if common_construct:
        warnings.append("Using shared endpoint construct rather than the canonical UniProt sequence")
        return SequenceChoice(canonical_sequence, common_construct, common_construct, "construct", warnings)

    warnings.append("Endpoint constructs differ; sequence selection requires manual review")
    return SequenceChoice(canonical_sequence, None, None, None, warnings)
