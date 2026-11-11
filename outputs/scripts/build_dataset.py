#!/usr/bin/env python3
"""Build a multi-conformation experimental protein dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from protein_conformer_dataset.config import DatasetConfig
from protein_conformer_dataset.logging_utils import configure_logging
from protein_conformer_dataset.pipeline import DatasetPipeline


def parse_args() -> argparse.Namespace:
    """Parse one mutually exclusive dataset input mode and run settings."""
    parser = argparse.ArgumentParser(
        description=(
            "Discover or process candidate experimental endpoint pairs. The output is QC evidence, "
            "not a claim of physical or kinetic connectivity."
        )
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--manifest", type=Path, help="CSV containing explicit endpoint pairs.")
    inputs.add_argument("--uniprot-ids", nargs="+", help="One or more UniProt accessions.")
    inputs.add_argument("--uniprot-ids-file", type=Path, help="Text file with one UniProt ID per line.")
    inputs.add_argument("--pdb-ids", nargs="+", help="PDB entries to group by exact UniProt mappings.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/dataset"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--top-k-pairs", type=int, default=3)
    parser.add_argument("--max-structures", type=int, default=50)
    parser.add_argument("--qc-mode", choices=("permissive", "strict"), default="permissive")
    parser.add_argument("--minimum-sequence-identity", type=float)
    parser.add_argument("--minimum-aligned-coverage", type=float)
    parser.add_argument("--minimum-common-ca-count", type=int)
    parser.add_argument("--maximum-resolution", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args()


def main() -> None:
    """Configure and run the selected dataset workflow."""
    args = parse_args()
    config = DatasetConfig.with_qc_mode(
        qc_mode=args.qc_mode,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        top_k_pairs=args.top_k_pairs,
        max_structures_per_protein=args.max_structures,
        resume=args.resume,
        force=args.force,
    )
    if args.minimum_sequence_identity is not None:
        config.qc.minimum_sequence_identity = args.minimum_sequence_identity
    if args.minimum_aligned_coverage is not None:
        config.qc.minimum_aligned_coverage = args.minimum_aligned_coverage
    if args.minimum_common_ca_count is not None:
        config.qc.minimum_common_ca_count = args.minimum_common_ca_count
    if args.maximum_resolution is not None:
        config.qc.maximum_resolution_angstrom = args.maximum_resolution

    logger = configure_logging(args.output_dir, getattr(logging, args.log_level))
    pipeline = DatasetPipeline(config, logger=logger)
    if args.manifest:
        summary = pipeline.run_manifest(args.manifest)
    elif args.uniprot_ids:
        summary = pipeline.run_uniprot_ids(args.uniprot_ids)
    elif args.uniprot_ids_file:
        identifiers = [
            line.strip()
            for line in args.uniprot_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        summary = pipeline.run_uniprot_ids(identifiers)
    else:
        summary = pipeline.run_pdb_ids(args.pdb_ids)
    print(summary.format())


if __name__ == "__main__":
    main()
