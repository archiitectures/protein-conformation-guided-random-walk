# Protein Conformer Dataset Package

This package builds reproducible datasets of candidate experimentally observed
protein endpoint pairs for downstream BioEmu sampling or graph construction. It
standardizes one protein chain per structure, aligns residues by amino-acid
sequence, computes matched C-alpha arrays, records QC evidence and environment
metadata, and exports everything needed for later manual review.

It does not identify a ground-truth transition. Multiple experimental
structures do not prove a direct transition pathway, equilibrium accessibility
does not imply kinetic connectivity, and high RMSD alone does not establish
meaningful conformational motion.

## Status Meanings

- `accepted`: passed the active numerical QC thresholds and triggered no review
  warnings.
- `manual_review`: passed the hard thresholds, but environment, construct,
  mutation, missing-residue, experimental-method, or oligomer/context metadata
  still needs human interpretation.
- `rejected`: failed at least one hard QC rule. Explicit manifest rows are still
  exported so their evidence remains inspectable.

`accepted` is not a claim of physical connectivity. It only means the pair
cleared the current filtering rules.

## File-By-File Python Map

- `__init__.py`: package exports.
- `alignment.py`: global sequence alignment, residue mapping, C-alpha matching,
  and Kabsch superposition.
- `checkpoint.py`: stage-level JSON checkpoint storage keyed by an
  input/configuration fingerprint.
- `config.py`: `DatasetConfig`, `QCConfig`, and `RankingConfig`.
- `download.py`: cache-aware HTTP client, retry policy, request pacing, and
  mmCIF download validation.
- `export.py`: per-pair export directory writing plus dataset-level CSV/YAML
  summary tables.
- `logging_utils.py`: console and `run_log.txt` logger setup.
- `models.py`: serializable structure, residue, alignment, QC, and pair models.
- `pair_selection.py`: metadata prefiltering, pair comparison, and candidate
  ranking.
- `parsing.py`: Gemmi-based chain parsing, unresolved-sequence preservation,
  altloc handling, and environment extraction.
- `pipeline.py`: top-level orchestration for UniProt discovery, PDB grouping,
  explicit manifests, checkpoint/resume, and export.
- `quality_control.py`: hard rejection rules plus manual-review warnings.
- `rcsb_client.py`: RCSB Search/Data API access and metadata normalization.
- `sequence.py`: residue-name normalization and BioEmu sequence selection.
- `sifts_client.py`: optional PDBe/SIFTS mapping fetcher for residue metadata.
- `structure.py`: writes processed protein-only C-alpha mmCIF files.
- `uniprot_client.py`: canonical UniProt sequence/identity metadata retrieval.

Related scripts:

- `scripts/build_dataset.py`: CLI entry point for dataset construction.
- `scripts/inspect_entry.py`: prints a concise per-entry QC summary from an
  exported manifest.

## Supported Input Modes

- `run_uniprot_ids(...)`: fetch canonical UniProt metadata, query exact
  UniProt-mapped RCSB polymer entities, compare the bounded candidate set, and
  export one selected pair per protein when ranking yields a usable result.
- `run_pdb_ids(...)`: resolve UniProt-mapped polymer entities from the supplied
  PDB entries, group them by exact UniProt accession, then follow the same
  automatic workflow as `run_uniprot_ids(...)`.
- `run_manifest(...)`: process each explicit CSV row independently. Required
  columns are `protein_id`, `pdb_a`, `chain_a`, `pdb_b`, and `chain_b`.
  Optional columns such as `uniprot_id`, `protein_name`, `state_a`, and
  `state_b` enrich the export and review tables.

## End-To-End `build_dataset.py` Flow

1. Parse one mutually exclusive input mode.
   `scripts/build_dataset.py` accepts `--manifest`, `--uniprot-ids`,
   `--uniprot-ids-file`, or `--pdb-ids`, plus output/cache/QC/resume options.

2. Build the run configuration.
   `DatasetConfig.with_qc_mode(...)` creates the default permissive or strict
   QC profile, then any CLI threshold overrides are applied.

3. Configure logging.
   `configure_logging(...)` writes both console output and
   `<output-dir>/run_log.txt`.

4. Create a checkpoint fingerprint.
   `DatasetPipeline` builds a `StateStore` whose fingerprint includes the input
   mode, the exact inputs, and the scientific configuration. `resume` and
   `force` are intentionally excluded from the fingerprint itself.

5. Resolve structures from the chosen input source.
   Automatic modes call UniProt and/or RCSB clients to discover candidate
   structures. Manifest mode resolves the exact requested `pdb/chain` pairs.

6. Metadata-prefilter large automatic structure sets before downloading.
   `prefilter_structure_records(...)` keeps at most
   `max_structures_per_protein` records, preferring constructs whose sequence
   length best matches the canonical UniProt length, then better resolution.

7. Reuse or write per-stage checkpoints.
   Each unit of work stores `metadata`, `downloads`, `parsing`, `comparisons`,
   and `export` payloads under `<output-dir>/.state/<unit>/`. A stage is reused
   only when `--resume` is enabled, the checkpoint is complete, and its
   fingerprint matches the current inputs/configuration.

8. Download raw coordinate files.
   `CachedHttpClient.download_cif(...)` fetches and validates RCSB mmCIF files
   into `<cache-dir>/structures/`. Corrupt or failed downloads are skipped
   without aborting the whole run.

9. Parse one protein chain per structure.
   `parse_mmcif_chain(...)` selects the requested chain, reads the first model's
   coordinates, preserves unresolved positions through the entity full sequence
   when available, chooses a single C-alpha altloc by occupancy, and records
   ligands, metals, nucleotides, partner chains, model count, and chain-count
   context.

10. Optionally map parsed residues to canonical UniProt positions.
   When a canonical sequence is known, `map_to_canonical(...)` aligns the parsed
   construct to UniProt and stores per-residue `uniprot_position` metadata.

11. Align endpoint sequences to each other.
   `align_chains(...)` performs a global amino-acid alignment, constructs a
   residue-by-residue mapping, and keeps only aligned positions where both
   endpoints have observed C-alpha atoms for coordinate arrays.

12. Compute structural superposition metrics.
   The matched C-alpha arrays are stored both in original coordinates and with
   endpoint A Kabsch-aligned into endpoint B's frame.

13. Evaluate QC.
   `evaluate_pair_quality(...)` checks sequence identity, aligned coverage,
   common C-alpha count, length mismatch, resolution, mutation annotations,
   missing-residue imbalance, experimental-method differences, oligomer/context
   metadata, ligands, partners, multi-model entries, and altloc handling.

14. Assign `accepted`, `manual_review`, or `rejected`.
   Rejection requires at least one hard-threshold failure. Any remaining warning
   without a rejection becomes `manual_review`.

15. Rank automatic candidate pairs.
   `rank_candidate_pairs(...)` excludes rejected pairs and near-duplicates below
   the duplicate-RMSD threshold, then scores the rest with a nonphysical
   heuristic using RMSD, coverage, identity, resolution, mutations, and missing
   residues. `top_k_pairs` only controls how many ranked rows are marked in the
   dataset-level CSV outputs; automatic export still writes only the single
   best-ranked pair directory.

16. Choose a BioEmu sequence.
   `choose_bioemu_sequence(...)` prefers the canonical UniProt sequence only
   when both endpoint constructs are sufficiently compatible with it. Otherwise
   it can fall back to a shared endpoint construct. If the endpoint constructs
   differ materially, no BioEmu sequence is selected and the ambiguity remains
   visible in the export.

17. Export the selected pair or explicit row.
   `export_endpoint_pair(...)` writes raw mmCIFs, processed protein-only mmCIFs,
   matched/aligned NumPy arrays, `common_residue_mapping.csv`, FASTA files,
   endpoint metadata JSON, ligand/environment metadata, a pair manifest, and a
   Markdown QC summary.

18. Write dataset-level tables.
   `write_dataset_tables(...)` writes `pairwise_metrics.csv`,
   `dataset_manifest.csv`, `top_candidate_pairs.csv`, status-specific CSVs,
   `all_structures.csv`, and `run_config.yaml`. `run_summary.json` records the
   aggregate counts and recoverable errors.

## Output Layout

Automatic discovery for one protein writes:

```text
<output-dir>/<protein_id>/
  manifest.json
  metadata/
  processed/
  raw/
  reports/
  sequence/
```

Explicit manifest mode writes each row under:

```text
<output-dir>/<protein_id>/pairs/<pair_id>/
```

The important files inside an exported pair directory are:

- `manifest.json`: top-level machine-readable description of the exported pair.
- `metadata/endpoint_A.json` and `metadata/endpoint_B.json`: normalized
  structure metadata for each endpoint.
- `metadata/pair_quality.json`: QC metrics, warnings, rejection reasons, and
  final status.
- `metadata/ligands.json`: ligand, metal, nucleotide, partner, and assembly
  context metadata.
- `processed/endpoint_A_ca.npy` and `processed/endpoint_B_ca.npy`: matched
  common C-alpha coordinates in original endpoint frames.
- `processed/endpoint_A_aligned_ca.npy` and
  `processed/endpoint_B_aligned_ca.npy`: the same coordinates after Kabsch
  superposition.
- `processed/common_residue_mapping.csv`: aligned residue correspondence with
  author numbering and optional UniProt positions.
- `processed/endpoint_A_protein_only.cif` and
  `processed/endpoint_B_protein_only.cif`: processed C-alpha-only protein files.
- `sequence/canonical.fasta`: canonical UniProt sequence when available.
- `sequence/construct.fasta`: shared endpoint construct sequence when both
  endpoints agree on it.
- `sequence/bioemu_selected.fasta`: the sequence currently judged safe to pass
  into BioEmu.
- `reports/qc_summary.md`: human-readable QC summary.

## Checkpoint and Resume Semantics

- `--resume` reuses only completed stage payloads whose fingerprint matches the
  current input list/manifest rows and scientific configuration.
- `--force` ignores both checkpoints and cached remote data.
- Automatic and explicit rows checkpoint independently, so one malformed entry
  does not invalidate the others.
- The exported directory must still exist for the `export` stage to count as
  reusable; the pipeline checks for the manifest file before skipping export.

## External Metadata Handling Caveats

- UniProt and RCSB metadata are cached locally but remain upstream-dependent.
- Sequence alignment, not author residue numbering, is the required residue
  correspondence mechanism.
- SIFTS support is best-effort enrichment only; the pipeline does not depend on
  SIFTS for the core mapping.
- `oligomeric_state`, partner-chain lists, and related QC warnings are derived
  from deposited chain content and simplified metadata. They are useful review
  signals, not authoritative biological-assembly truth.
- Ligand, metal, nucleotide, membrane, and partner context is preserved because
  environment changes can explain apparent structural differences.

## Scientific Boundaries

- Candidate pairs are not validated physical transitions.
- Equilibrium accessibility does not imply kinetic connectivity.
- Ligand-bound and apo structures can represent different environments.
- Mutations and truncations can create artificial structural differences.
- Oligomeric changes need treatment distinct from monomeric motion.
- Missing residues can distort RMSD and apparent coverage.
- High RMSD can reflect artifacts, incomplete overlap, or context changes.
- BioEmu should only receive a sequence consistent with both endpoint
  constructs.
- The package exports candidate endpoints and QC evidence, never ground truth.
