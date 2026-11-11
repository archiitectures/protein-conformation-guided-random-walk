"""Configurable endpoint-pair quality control and scientific warnings.

Scientific invariants enforced by this package:

- Multiple structures do not prove a direct transition pathway.
- Equilibrium accessibility does not imply kinetic connectivity.
- Ligand-bound and apo structures may represent different environments.
- Mutations and truncations can create artificial structural differences.
- Oligomeric changes require treatment distinct from monomeric motion.
- Missing residues can inflate or distort RMSD.
- PDB author numbering is metadata, not an alignment key.
- BioEmu must receive a sequence consistent with both endpoint constructs.
- Candidate pairs plus QC evidence are not ground-truth transition labels.
"""

from __future__ import annotations

from protein_conformer_dataset.alignment import ChainAlignment
from protein_conformer_dataset.config import QCConfig
from protein_conformer_dataset.models import PairQualityReport, PairStatus, ParsedChain


def evaluate_pair_quality(
    chain_a: ParsedChain,
    chain_b: ParsedChain,
    alignment: ChainAlignment,
    config: QCConfig,
) -> PairQualityReport:
    """Classify a pair from sequence, coordinate, and environment evidence.

    Multiple experimental conformations do not prove direct kinetic
    connectivity. This function only decides whether a pair is sufficiently
    consistent to be retained as a candidate endpoint comparison.
    """
    record_a = chain_a.record
    record_b = chain_b.record
    warnings: list[str] = []
    rejection_reasons: list[str] = []
    metrics = alignment.metrics
    common_ca_count = len(alignment.coords_a)
    length_difference = abs(len(chain_a.residues) - len(chain_b.residues))
    length_denominator = max(len(chain_a.residues), len(chain_b.residues), 1)
    length_difference_fraction = length_difference / length_denominator
    missing_a = 1.0 - chain_a.ca_count / max(len(chain_a.residues), 1)
    missing_b = 1.0 - chain_b.ca_count / max(len(chain_b.residues), 1)

    same_uniprot = bool(
        record_a.uniprot_id
        and record_b.uniprot_id
        and record_a.uniprot_id == record_b.uniprot_id
    )
    same_organism = _optional_equal(record_a.organism, record_b.organism)
    oligomeric_match = _optional_equal(record_a.oligomeric_state, record_b.oligomeric_state)

    if record_a.uniprot_id != record_b.uniprot_id:
        warnings.append("UniProt identifiers differ or are unavailable")
    if same_organism is False:
        warnings.append("Source organisms differ")
    if metrics.identity < config.minimum_sequence_identity:
        rejection_reasons.append(
            f"Sequence identity {metrics.identity:.3f} is below {config.minimum_sequence_identity:.3f}"
        )
    if min(metrics.coverage_a, metrics.coverage_b) < config.minimum_aligned_coverage:
        rejection_reasons.append(
            "Aligned sequence coverage is below " f"{config.minimum_aligned_coverage:.3f}"
        )
    if common_ca_count < config.minimum_common_ca_count:
        rejection_reasons.append(
            f"Only {common_ca_count} common C-alpha atoms; minimum is {config.minimum_common_ca_count}"
        )
    if length_difference_fraction > config.maximum_length_difference_fraction:
        rejection_reasons.append(
            f"Chain length difference fraction {length_difference_fraction:.3f} exceeds "
            f"{config.maximum_length_difference_fraction:.3f}"
        )
    for label, record in (("A", record_a), ("B", record_b)):
        if record.resolution is not None and record.resolution > config.maximum_resolution_angstrom:
            rejection_reasons.append(
                f"Endpoint {label} resolution {record.resolution:.2f} A exceeds "
                f"{config.maximum_resolution_angstrom:.2f} A"
            )
    mutation_count = metrics.mutation_count + len(record_a.mutation_annotations) + len(
        record_b.mutation_annotations
    )
    if mutation_count:
        warnings.append(f"Detected {mutation_count} sequence mismatch/mutation annotation(s)")
        if config.exclude_engineered_mutations:
            rejection_reasons.append("Engineered mutations are excluded by the active QC profile")
    if length_difference:
        warnings.append(f"Endpoint chain lengths differ by {length_difference} residues")
    if abs(missing_a - missing_b) > config.missing_pattern_difference_threshold:
        warnings.append("Missing-residue fractions differ substantially and may distort RMSD")
    method_a = (record_a.experimental_method or "").upper()
    method_b = (record_b.experimental_method or "").upper()
    nmr_a = "NMR" in method_a
    nmr_b = "NMR" in method_b
    if nmr_a != nmr_b:
        warnings.append("One endpoint is NMR and the other uses a different experimental method")
    if config.exclude_nmr and (nmr_a or nmr_b):
        rejection_reasons.append("NMR structures are excluded by the active QC profile")
    if oligomeric_match is False:
        warnings.append("Deposited protein-chain counts/oligomeric states differ")
        if config.exclude_multimeric_changes:
            rejection_reasons.append("Oligomeric-state changes are excluded by the active QC profile")
    if set(record_a.partner_chain_ids) != set(record_b.partner_chain_ids):
        warnings.append("Protein-binding partner chains differ between endpoints")
    if set(record_a.nucleic_acid_partner_chain_ids) != set(record_b.nucleic_acid_partner_chain_ids):
        warnings.append("DNA/RNA partner chains differ between endpoints")
    if record_a.membrane_associated != record_b.membrane_associated and (
        record_a.membrane_associated is not None or record_b.membrane_associated is not None
    ):
        warnings.append("Membrane-association metadata differs between endpoints")
    if _environment_set(record_a) != _environment_set(record_b):
        warnings.append("Ligand, metal, or nucleotide occupancy differs between endpoints")
    if record_a.model_count > 1 or record_b.model_count > 1:
        warnings.append("At least one endpoint contains multiple experimental models")
    if record_a.alternate_locations_present or record_b.alternate_locations_present:
        warnings.append("Alternate atom locations were present and occupancy-selected")
    if min(missing_a, missing_b) > 0.15:
        warnings.append("At least one endpoint has substantial unresolved sequence")

    if rejection_reasons:
        status = PairStatus.REJECTED
    elif warnings:
        status = PairStatus.MANUAL_REVIEW
    else:
        status = PairStatus.ACCEPTED
    return PairQualityReport(
        sequence_identity=metrics.identity,
        aligned_coverage_a=metrics.coverage_a,
        aligned_coverage_b=metrics.coverage_b,
        common_ca_count=common_ca_count,
        raw_ca_rmsd=alignment.raw_rmsd,
        ca_rmsd=alignment.aligned_rmsd,
        length_difference=length_difference,
        mutation_count=mutation_count,
        missing_fraction_a=missing_a,
        missing_fraction_b=missing_b,
        same_uniprot=same_uniprot,
        same_organism=same_organism,
        oligomeric_state_match=oligomeric_match,
        warnings=warnings,
        rejection_reasons=rejection_reasons,
        status=status,
    )


def _optional_equal(value_a: object | None, value_b: object | None) -> bool | None:
    if value_a is None or value_b is None:
        return None
    return value_a == value_b


def _environment_set(record) -> set[str]:
    return set(record.ligand_ids) | set(record.metal_ids) | set(record.nucleotide_ids)
