#!/usr/bin/env python3
"""Print the QC evidence and paths for one exported endpoint pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """Parse inspection arguments."""
    parser = argparse.ArgumentParser(description="Inspect an exported conformer dataset entry.")
    parser.add_argument("--protein-id", required=True)
    parser.add_argument("--pair-id", help="Required when an explicit protein has multiple pair folders.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/dataset"))
    return parser.parse_args()


def locate_manifest(dataset_dir: Path, protein_id: str, pair_id: str | None = None) -> Path:
    """Locate an automatic or explicit-pair manifest."""
    protein_dir = dataset_dir / protein_id
    direct = protein_dir / "manifest.json"
    if pair_id:
        path = protein_dir / "pairs" / pair_id / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"Pair manifest not found: {path}")
        return path
    if direct.exists():
        return direct
    candidates = sorted((protein_dir / "pairs").glob("*/manifest.json"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No exported entry found for protein ID {protein_id!r}")
    available = ", ".join(path.parent.name for path in candidates)
    raise ValueError(f"Multiple pairs exist; pass --pair-id. Available: {available}")


def format_manifest(manifest: dict[str, Any], manifest_path: Path) -> str:
    """Format the most important metadata and QC evidence for manual review."""
    quality = manifest["quality"]
    endpoint_a = manifest["structure_a"]
    endpoint_b = manifest["structure_b"]
    sequence = manifest.get("sequence_selection", {})
    lines = [
        f"Protein: {manifest['protein_id']}",
        f"Pair: {manifest['pair_id']}",
        f"Status: {quality['status']}",
        f"Endpoints: {endpoint_a['pdb_id']} chain {endpoint_a['auth_chain_id']} -> "
        f"{endpoint_b['pdb_id']} chain {endpoint_b['auth_chain_id']}",
        f"UniProt: {manifest.get('uniprot_id')}",
        f"Sequence identity: {quality['sequence_identity']:.4f}",
        f"Aligned coverage: {min(quality['aligned_coverage_a'], quality['aligned_coverage_b']):.4f}",
        f"Common C-alpha count: {quality['common_ca_count']}",
        f"Aligned C-alpha RMSD: {quality['ca_rmsd']}",
        f"Missing fractions: A={quality['missing_fraction_a']:.4f}, "
        f"B={quality['missing_fraction_b']:.4f}",
        f"Ligands: A={endpoint_a['ligand_ids']}, B={endpoint_b['ligand_ids']}",
        f"Protein partners: A={endpoint_a['partner_chain_ids']}, B={endpoint_b['partner_chain_ids']}",
        f"Mutations/mismatches: {quality['mutation_count']}",
        f"BioEmu sequence source: {sequence.get('source')}",
        f"Manifest: {manifest_path}",
        "Warnings:",
    ]
    lines.extend(f"  - {warning}" for warning in quality.get("warnings", []))
    if not quality.get("warnings"):
        lines.append("  - None")
    lines.append("Rejection reasons:")
    lines.extend(f"  - {reason}" for reason in quality.get("rejection_reasons", []))
    if not quality.get("rejection_reasons"):
        lines.append("  - None")
    return "\n".join(lines)


def main() -> None:
    """Load and print one entry manifest."""
    args = parse_args()
    path = locate_manifest(args.dataset_dir, args.protein_id, args.pair_id)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    print(format_manifest(manifest, path))


if __name__ == "__main__":
    main()
