# Tests Walkthrough

## Experimental Dataset Tests

The `test_dataset_*` modules use synthetic mmCIF fixtures and mocked clients, so
the normal suite never requires internet access. They cover global sequence
alignment, sequence identity and mutations, missing residues, insertion codes,
altloc occupancy, multiple models, ligands and partner chains, Kabsch
superposition, QC rejection/manual-review behavior, candidate ranking,
standardized export, explicit manifest mode, HTTP caching, corrupt downloads,
and stage-level resume.

This folder contains pytest coverage for package behavior and experiment helpers.

Run tests with:

```bash
UV_CACHE_DIR=.uv-cache uv run pytest
```

## `test_scoring.py`

Covers scoring behavior.

Key checks:

- Kabsch RMSD is approximately zero for rotated/translated identical coordinates.
- RMSD rejects mismatched coordinate lengths.
- Distance histogram scorer returns finite non-negative values.
- Distance histograms are normalized.

These tests protect the assumption that endpoint coordinate arrays must already represent corresponding residues.

## `test_validity.py`

Covers C-alpha validity checks.

Key checks:

- NaN/inf coordinates are rejected.
- Unrealistic consecutive C-alpha distances are penalized.
- Non-neighbor C-alpha clashes are detected.
- Bonded neighbors are not treated as clashes.

Reminder: these are coarse C-alpha checks, not full atomistic validation.

## `test_generators.py`

Covers node generation.

Key checks:

- Dummy Gaussian generator returns expected shapes/source labels.
- Invalid/clashing proposals can be filtered.
- Validity filtering can be disabled.

These tests reflect the current policy: generator requests are targets, not guarantees, because validity filtering may return fewer nodes.

## `test_graph.py`

Covers graph construction, path search, and graph visualization.

Key checks:

- Small fully connected graphs have expected node/edge counts.
- Shortest paths start at A and end at B.
- `plot_graph(...)` can write sparse/faded edge views.

## `test_large_synthetic_experiment.py`

Covers the reusable large experiment module.

Key areas:

- Synthetic globular endpoint generation.
- Recursive Gaussian node summary metadata and frontier selection.
- `allow_fallback=False` behavior for keeping recursive-only results.
- Fallback route graph construction.
- Route-family cost reporting.
- PyMOL path-view output.
- py3Dmol HTML and notebook viewer helpers.
- PDB endpoint loading.
- Exact residue matching.
- Geometric fallback matching.

## `test_bioemu_integration.py`

Covers the BioEmu bundle workflow without requiring BioEmu itself.

Key areas:

- FASTA parsing.
- Portable bundle manifest loading.
- Local precomputed BioEmu graph analysis.
- CLI flag parsing for the BioEmu analyzer.
- Fake-sampler bundle normalization through `run_bioemu_from_fasta(...)`.
- Resolution of the documented `bioemu.sample.main` entry point.
- Expansion of simulated official `topology.pdb`/`samples.xtc` output into one
  Angstrom-scale C-alpha node per frame without installing BioEmu locally.

## `test_dataset_bioemu_adapter.py`

Covers the automatic dataset-pair to Colab BioEmu handoff.

Key areas:

- Full-sequence BioEmu samples are subset to common endpoint residue positions.
- Exact bundle/dataset sequence validation.
- Out-of-range residue mapping rejection.
- End-to-end adapted graph construction and output writing.
- Dataset-aware CLI flag parsing.

`test_validity.py` additionally verifies that missing-residue gaps are not
treated as C-alpha bonds and remain eligible for non-neighbor clash checks.

## `test_colab_transfer.py`

Covers deterministic workspace packaging, five-pair validation, result archive
round trips, overwrite protection, archive path traversal rejection, and CLI
parsing.

## `test_colab_notebooks.py`

Enforces the L4-only, no-Drive notebook contract, all five production pair
calls, canonical local post-download paths, and Python syntax for every BioEmu
notebook code cell.

## Adding Tests

Prefer small tests around reusable package functions.

For notebook-facing features:

1. Put core behavior in `guided_conformational_path/`.
2. Test the package function.
3. Keep notebooks as UI only.

Avoid tests that depend on a local GUI, PyMOL installation, browser, or internet access.
