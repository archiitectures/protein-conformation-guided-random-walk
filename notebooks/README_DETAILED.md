# Notebooks Walkthrough

This folder contains notebook-based user interfaces. The notebook should remain a UI layer over package functions.

## Environment Setup

Install the notebook extras with:

```bash
UV_CACHE_DIR=.uv-cache uv sync --extra test --extra notebook
```

Optional kernel registration:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m ipykernel install --user --name guided-conformational-path
```

Then open:

```text
notebooks/large_synthetic_experiment.ipynb
```

For BioEmu sampling on Colab, open:

```text
notebooks/run_bioemu_colab.ipynb
```

For local analysis of a downloaded BioEmu bundle, open:

```text
notebooks/analyze_bioemu_outputs.ipynb
```

For a step-by-step Colab-to-local graph workflow, open:

```text
notebooks/bioemu_colab_to_graph_workflow.ipynb
```

## `large_synthetic_experiment.ipynb`

This notebook supports two workflows:

1. Generated synthetic A/B endpoints.
2. PDB-provided A/B endpoints.

It imports reusable logic from:

```python
guided_conformational_path.large_synthetic
```

The notebook should not contain independent implementations of generation, scoring, matching, or graph construction.

## Synthetic Endpoint Workflow

The synthetic section uses `LargeSyntheticConfig`.

Important parameters:

- `n_residues`: number of C-alpha points.
- `n_samples`: total accepted recursive Gaussian node cap across both sides.
- `noise_scale`: Gaussian perturbation magnitude.
- `recursive_max_iterations`: maximum recursive expansion rounds.
- `recursive_top_k`: closest cross-side frontier parents retained per side.
- `recursive_samples_per_parent`: Gaussian children requested per selected parent.
- `allow_fallback`: disable this to inspect the recursive Gaussian path directly even when it misses the target path shape.
- `fallback_waypoints`: number of route-family waypoints.
- `fallback_samples_per_waypoint`: optional Gaussian samples around waypoints.
- `k_neighbors`: sparse graph construction control.
- `write_animation`: writes the best-path morph GIF.
- `animation_frames_per_edge`: interpolated frames per path edge.
- `animation_fps`: GIF playback speed.
- `animation_dpi`: render quality for the GIF frames.
- `graph_max_edge_weight`: sparse display control for hiding high-cost non-path edges.
- `graph_non_path_edge_alpha`: transparency for non-path edges in `graph.png`.

The synthetic A/B endpoints are compact globular C-alpha traces, not real proteins.

Notebook progress note:

- The recursive sampler accepts a `progress_callback`.
- `large_synthetic_experiment.ipynb` uses that hook to drive tqdm plus a visible notebook fallback status block.

## PDB Endpoint Workflow

Drop files here:

```text
data/pdb_inputs/A.pdb
data/pdb_inputs/B.pdb
```

The notebook uses `PDBEndpointConfig`.

Matching controls:

- `matching_method="exact_then_geometric"`
- `min_exact_match_fraction=0.8`
- `geometric_match_distance_cutoff=5.0`
- `min_geometric_matches=8`

Exact matching uses:

```text
chain ID + residue number + insertion code
```

Geometric fallback:

- aligns C-alpha coordinates with Kabsch using an initial prefix pairing
- matches mutual nearest C-alpha neighbors
- keeps only matches inside the cutoff

This is heuristic. It is useful for similar-backbone exploratory tests, not rigorous sequence/structure alignment.

## Notebook Outputs

Typical output folders:

```text
outputs/notebook_large_synthetic/
outputs/pdb_endpoint_experiment/
```

Important files:

- `experiment_summary.json`: full run metadata.
- `path_summary.json`: shortest path record.
- `graph.png`: NetworkX graph view.
- `path_morph.gif`: best-path morph animation.
- `py3dmol_path_view.html`: browser-based 3D path overlay.
- `pymol_path/view_path.pml`: PyMOL script.

## BioEmu Colab Workflow

`run_bioemu_colab.ipynb` is intentionally thin:

1. Checks GPU availability.
2. Installs `bioemu[cuda]` and this package.
3. Mounts Google Drive.
4. Reads a FASTA file.
5. Calls `run_bioemu_from_fasta(...)`.
6. Writes a portable bundle with `manifest.json`, `nodes/*.pdb`, and raw outputs.

Download or sync that bundle to `data/bioemu_outputs/latest`, then run the local
precomputed BioEmu CLI to build the graph and path outputs.

## py3Dmol View

`display_py3dmol_path(run)` overlays A, B, and path nodes.

It does not show the NetworkX graph. It shows the C-alpha structures associated with the selected graph path.

## Notebook Hygiene

Before committing notebook changes, prefer clearing large outputs if they are not needed for documentation.

Keep exploratory parameter changes visible and intentional.
