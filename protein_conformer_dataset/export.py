"""Standardized per-pair and dataset-level export functions."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from protein_conformer_dataset.models import EndpointPair, ParsedChain, StructureRecord
from protein_conformer_dataset.sequence import SequenceChoice
from protein_conformer_dataset.structure import write_protein_only_cif


MANIFEST_COLUMNS = [
    "protein_id", "protein_name", "uniprot_id", "pdb_a", "chain_a", "pdb_b", "chain_b",
    "state_a", "state_b", "sequence_identity", "aligned_coverage", "common_ca_count",
    "ca_rmsd", "raw_ca_rmsd", "experimental_method_a", "experimental_method_b",
    "resolution_a", "resolution_b", "ligands_a", "ligands_b", "mutation_count",
    "candidate_score", "candidate_rank", "status", "accepted", "warnings", "rejection_reasons",
    "output_path",
]


def export_endpoint_pair(
    pair: EndpointPair,
    chain_a: ParsedChain,
    chain_b: ParsedChain,
    sequence_choice: SequenceChoice,
    destination: str | Path,
    *,
    domain_selection: dict[str, Any] | None = None,
    residue_selection: dict[str, Any] | None = None,
) -> Path:
    """Write one complete endpoint-pair directory and return its manifest path."""
    root = Path(destination)
    for child in ("sequence", "raw", "processed", "metadata", "reports"):
        (root / child).mkdir(parents=True, exist_ok=True)

    _copy_raw(pair.structure_a, root / "raw")
    _copy_raw(pair.structure_b, root / "raw")
    write_protein_only_cif(root / "processed" / "endpoint_A_protein_only.cif", chain_a)
    write_protein_only_cif(root / "processed" / "endpoint_B_protein_only.cif", chain_b)
    np.save(root / "processed" / "endpoint_A_ca.npy", pair.coords_a)
    np.save(root / "processed" / "endpoint_B_ca.npy", pair.coords_b)
    np.save(root / "processed" / "endpoint_A_aligned_ca.npy", pair.aligned_coords_a)
    np.save(root / "processed" / "endpoint_B_aligned_ca.npy", pair.aligned_coords_b)
    _write_mapping(root / "processed" / "common_residue_mapping.csv", pair)

    if sequence_choice.canonical_sequence:
        _write_fasta(
            root / "sequence" / "canonical.fasta",
            pair.uniprot_id or pair.protein_id,
            sequence_choice.canonical_sequence,
        )
    if sequence_choice.construct_sequence:
        _write_fasta(
            root / "sequence" / "construct.fasta",
            f"{pair.protein_id}_construct",
            sequence_choice.construct_sequence,
        )
    if sequence_choice.selected_sequence:
        _write_fasta(
            root / "sequence" / "bioemu_selected.fasta",
            f"{pair.protein_id}_{sequence_choice.source}",
            sequence_choice.selected_sequence,
        )

    _write_json(root / "metadata" / "endpoint_A.json", pair.structure_a.to_dict())
    _write_json(root / "metadata" / "endpoint_B.json", pair.structure_b.to_dict())
    _write_json(root / "metadata" / "pair_quality.json", pair.quality.to_dict())
    _write_json(
        root / "metadata" / "ligands.json",
        {
            "endpoint_A": _environment_metadata(pair.structure_a),
            "endpoint_B": _environment_metadata(pair.structure_b),
        },
    )
    manifest = pair.to_dict(include_coordinates=False)
    manifest["sequence_selection"] = asdict(sequence_choice)
    if domain_selection is not None:
        manifest["domain_selection"] = domain_selection
    if residue_selection is not None:
        manifest["residue_selection"] = residue_selection
    manifest["scientific_caveat"] = (
        "This is a candidate experimentally observed endpoint pair. Multiple structures do not prove "
        "a direct transition, and equilibrium accessibility does not imply kinetic connectivity."
    )
    manifest["paths"] = {
        "endpoint_a_ca": "processed/endpoint_A_ca.npy",
        "endpoint_b_ca": "processed/endpoint_B_ca.npy",
        "endpoint_a_aligned_ca": "processed/endpoint_A_aligned_ca.npy",
        "endpoint_b_aligned_ca": "processed/endpoint_B_aligned_ca.npy",
        "bioemu_sequence": (
            "sequence/bioemu_selected.fasta" if sequence_choice.selected_sequence else None
        ),
    }
    manifest_path = root / "manifest.json"
    _write_json(manifest_path, manifest)
    (root / "reports" / "qc_summary.md").write_text(
        _qc_markdown(
            pair,
            sequence_choice,
            domain_selection=domain_selection,
            residue_selection=residue_selection,
        ),
        encoding="utf-8",
    )
    return manifest_path


def pair_table_row(pair: EndpointPair, output_path: str | Path | None = None) -> dict[str, Any]:
    """Flatten endpoint evidence for dataset-level CSV tables."""
    quality = pair.quality
    return {
        "protein_id": pair.protein_id,
        "protein_name": pair.structure_a.protein_name or pair.structure_b.protein_name,
        "uniprot_id": pair.uniprot_id,
        "pdb_a": pair.structure_a.pdb_id,
        "chain_a": pair.structure_a.auth_chain_id,
        "pdb_b": pair.structure_b.pdb_id,
        "chain_b": pair.structure_b.auth_chain_id,
        "state_a": pair.state_a,
        "state_b": pair.state_b,
        "sequence_identity": quality.sequence_identity,
        "aligned_coverage": min(quality.aligned_coverage_a, quality.aligned_coverage_b),
        "common_ca_count": quality.common_ca_count,
        "ca_rmsd": quality.ca_rmsd,
        "raw_ca_rmsd": quality.raw_ca_rmsd,
        "experimental_method_a": pair.structure_a.experimental_method,
        "experimental_method_b": pair.structure_b.experimental_method,
        "resolution_a": pair.structure_a.resolution,
        "resolution_b": pair.structure_b.resolution,
        "ligands_a": ";".join(pair.structure_a.ligand_ids),
        "ligands_b": ";".join(pair.structure_b.ligand_ids),
        "mutation_count": quality.mutation_count,
        "candidate_score": pair.candidate_score,
        "candidate_rank": None,
        "status": quality.status.value,
        "accepted": quality.accepted,
        "warnings": " | ".join(quality.warnings),
        "rejection_reasons": " | ".join(quality.rejection_reasons),
        "output_path": str(output_path) if output_path else None,
    }


def structure_table_row(record: StructureRecord) -> dict[str, Any]:
    """Flatten structure metadata for ``all_structures.csv``."""
    data = record.to_dict()
    for key in (
        "ligand_ids", "metal_ids", "nucleotide_ids", "partner_chain_ids",
        "nucleic_acid_partner_chain_ids", "mutation_annotations",
    ):
        data[key] = ";".join(data[key])
    data["metadata"] = json.dumps(data["metadata"], sort_keys=True)
    return data


def write_dataset_tables(
    output_dir: str | Path,
    pair_rows: Iterable[dict[str, Any]],
    structure_rows: Iterable[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    """Write reproducible dataset-level manifests and status-specific tables."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = list(pair_rows)
    structures = list(structure_rows)
    dataframe = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    dataframe.to_csv(root / "pairwise_metrics.csv", index=False)
    dataframe[dataframe["output_path"].notna()].to_csv(root / "dataset_manifest.csv", index=False)
    dataframe[dataframe["candidate_rank"].notna()].to_csv(root / "top_candidate_pairs.csv", index=False)
    dataframe[dataframe["status"] == "accepted"].to_csv(root / "accepted_pairs.csv", index=False)
    dataframe[dataframe["status"] == "manual_review"].to_csv(
        root / "manual_review_pairs.csv", index=False
    )
    dataframe[dataframe["status"] == "rejected"].to_csv(root / "rejected_pairs.csv", index=False)
    pd.DataFrame(structures).to_csv(root / "all_structures.csv", index=False)
    (root / "run_config.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")


def _copy_raw(record: StructureRecord, destination: Path) -> None:
    if not record.file_path:
        return
    source = Path(record.file_path)
    if source.exists():
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination / f"{record.pdb_id}.cif")


def _write_mapping(path: Path, pair: EndpointPair) -> None:
    fieldnames = list(asdict(pair.residue_mapping[0]).keys()) if pair.residue_mapping else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(asdict(item) for item in pair.residue_mapping)


def _write_fasta(path: Path, record_id: str, sequence: str) -> None:
    wrapped = "\n".join(sequence[index : index + 80] for index in range(0, len(sequence), 80))
    path.write_text(f">{record_id}\n{wrapped}\n", encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _environment_metadata(record: StructureRecord) -> dict[str, Any]:
    return {
        "ligand_ids": record.ligand_ids,
        "metal_ids": record.metal_ids,
        "nucleotide_ids": record.nucleotide_ids,
        "protein_partner_chain_ids": record.partner_chain_ids,
        "nucleic_acid_partner_chain_ids": record.nucleic_acid_partner_chain_ids,
        "membrane_associated": record.membrane_associated,
        "biological_assembly_id": record.biological_assembly_id,
        "oligomeric_state": record.oligomeric_state,
        "ph": record.ph,
        "temperature": record.temperature,
    }


def _qc_markdown(
    pair: EndpointPair,
    sequence_choice: SequenceChoice,
    *,
    domain_selection: dict[str, Any] | None = None,
    residue_selection: dict[str, Any] | None = None,
) -> str:
    quality = pair.quality
    warning_lines = "\n".join(f"- {item}" for item in quality.warnings) or "- None"
    rejection_lines = "\n".join(f"- {item}" for item in quality.rejection_reasons) or "- None"
    sequence_lines = "\n".join(f"- {item}" for item in sequence_choice.warnings) or "- None"
    domain_section = ""
    if domain_selection is not None:
        domain_section = f"""
## Domain Selection

- Mode: `{domain_selection.get('mode', 'unknown')}`
- Selected residues: {domain_selection.get('selected_residue_count', 'unknown')}
- Caveat: {domain_selection.get('caveat', 'Domain-only comparison')}
"""
    residue_section = ""
    if residue_selection is not None:
        residue_section = f"""
## Residue Selection

- Mode: `{residue_selection.get('mode', 'unknown')}`
- Selected residues: {residue_selection.get('selected_residue_count', 'unknown')}
- Caveat: {residue_selection.get('caveat', 'Only selected matched residues are compared.')}
"""
    return f"""# Endpoint Pair QC: {pair.pair_id}

Status: **{quality.status.value}**

- Sequence identity: {quality.sequence_identity:.4f}
- Aligned coverage: {min(quality.aligned_coverage_a, quality.aligned_coverage_b):.4f}
- Common C-alpha atoms: {quality.common_ca_count}
- Aligned C-alpha RMSD: {quality.ca_rmsd}
- Candidate selection score: {pair.candidate_score:.4f}
- BioEmu sequence source: {sequence_choice.source}

## Warnings

{warning_lines}

## Rejection Reasons

{rejection_lines}

## Sequence Selection Warnings

{sequence_lines}

## Scientific Caveat

Multiple experimental structures do not prove a direct transition pathway. Equilibrium
accessibility does not imply kinetic connectivity. Ligands, mutations, truncations,
oligomeric changes, and missing residues can all produce structural differences. This
report identifies a candidate endpoint pair plus QC evidence, not ground truth.
{domain_section}
{residue_section}
"""
