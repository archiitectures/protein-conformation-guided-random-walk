# Tests Walkthrough

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

## Adding Tests

Prefer small tests around reusable package functions.

For notebook-facing features:

1. Put core behavior in `guided_conformational_path/`.
2. Test the package function.
3. Keep notebooks as UI only.

Avoid tests that depend on a local GUI, PyMOL installation, browser, or internet access.
