# Notebooks Walkthrough

This folder contains notebook-based user interfaces. The notebook should remain a UI layer over package functions.

## Environment Setup

Install the notebook extras with:

```bash
uv sync --extra test --extra notebook
```

Optional kernel registration:

```bash
uv run python -m ipykernel install --user --name guided-conformational-path --display-name "guided-conformational-path (.venv)"
```

Select the `guided-conformational-path (.venv)` kernel in VS Code or Jupyter.
If `tqdm` or another dependency is missing, the active notebook kernel is not the
project `.venv`.

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
- `large_synthetic_experiment.ipynb` uses that hook to drive tqdm plus a visible notebook status block.

## PDB Endpoint Workflow

Drop files here:

```text
data/pdb_inputs/a.pdb
data/pdb_inputs/b.pdb
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

`run_bioemu_colab.ipynb` is the executable no-Drive first-five workflow:

1. Requires the official VS Code Colab extension and accepts NVIDIA `L4` or
   `T4` runtimes by default.
2. Reports the connected GPU plus allowed list and stops rather than falling
   back to another device.
3. Extracts the locally created workspace ZIP uploaded to `/content`.
4. Installs `bioemu[cuda]` and the uploaded project in the remote runtime.
5. Requests two smoke samples for `P69441 / 4AKE_A__1AKE_A`.
6. Provides separate 200-sample cells for all five pairs, ordered by sequence
   length.
7. Converts BioEmu `topology.pdb` plus `samples.xtc` into portable PDB nodes.
8. Runs the residue adapter, graph construction, path search, and visual output
   writers after every completed bundle.
9. Creates one result ZIP per pair under `/content/exports` for immediate
   extension download.

The notebook file and canonical data remain local, but its selected kernel and
`/content` filesystem are remote and ephemeral. Do not use Run All for the
production cells; download and import each result before beginning the next.
Each production cell remains one pair/sample-count task because BioEmu still
lacks mid-sample checkpointing.

`bioemu_colab_to_graph_workflow.ipynb` is now a local post-download verifier. It
reports import status for all five production pairs and can rerun graph analysis
without BioEmu or a GPU.

`analyze_bioemu_outputs.ipynb` is the concise local interface for a downloaded
dataset-aware bundle. The older standalone endpoint analyzer remains
available through `scripts/run_bioemu_precomputed_experiment.py`.

Every BioEmu task distinguishes `requested_samples` from `retained_samples`.
The first is the requested generation target; the second is the number of
validated bundle nodes actually available to graph analysis. A task can retain
fewer nodes after trajectory conversion or validity filtering.

## py3Dmol View

`display_py3dmol_path(run)` overlays A, B, and path nodes.

It does not show the NetworkX graph. It shows the C-alpha structures associated with the selected graph path.

## Notebook Hygiene

Before committing notebook changes, prefer clearing large outputs if they are not needed for documentation.

Keep exploratory parameter changes visible and intentional.
