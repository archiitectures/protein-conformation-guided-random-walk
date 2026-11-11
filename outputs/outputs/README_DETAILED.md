# Outputs Folder Walkthrough

This folder is for generated run artifacts. Most generated contents are ignored by Git.

## Common Files

Quick demo outputs:

```text
outputs/path_summary.json
outputs/graph.png
outputs/nodes/
```

Large synthetic experiment outputs:

```text
outputs/large_synthetic/
  experiment_summary.json
  path_summary.json
  graph.png
  path_morph.gif
  path_gifs/
    top_exact_01_global_001.gif
    top_diverse_01_global_<rank>.gif
    random_non_top_01.gif
  py3dmol_path_view.html
  pymol_path/
    view_path.pml
    00_A.pdb
    ...
```

Runs that enable candidate reporting also write `candidate_paths.json` and
`candidate_paths.html`.

The JSON and sortable HTML reports contain the bounded exact top-K paths,
diversity-filtered alternatives, and any selected `random_non_top` comparison
paths. They include costs, deltas, edge overlap/diversity, intermediate-node
overlap, and per-edge diagnostics; the HTML adds a shared-layout graph thumbnail
for every row. The default configuration requests four exact paths and up to
four alternatives with at least 20% edge-set diversity from the best path.
These are heuristic graph comparisons, not kinetic or probability reports.

## Path GIFs

`path_morph.gif` remains the canonical best-path animation. With animation
enabled, the default categorized set is written under `path_gifs/`:

- `num_top_gifs_generated=6` splits into three exact and three diverse paths.
- `num_bottom_gifs_generated=1` selects one deterministic non-top simple path,
  excluding the best path and the exact/diverse report union.
- Exact/diverse names are
  `top_exact_<category-rank>_global_<cost-rank>.gif` and
  `top_diverse_<category-rank>_global_<cost-rank>.gif`, with zero-padded ranks.
- Non-top names are `random_non_top_<category-rank>.gif`.

Setting either count to zero disables that category; negative counts are
invalid. Counts are upper bounds when fewer qualifying paths are available.
Disabling animation suppresses both `path_morph.gif` and `path_gifs/`.

Before each output rerun, stale feature-owned categorized files matching the
`top_exact_*`, `top_diverse_*`, and `random_non_top_*` GIF naming schemes are
removed. Other files in `path_gifs/`, and unrelated files elsewhere in the
output directory, are left intact.

GIF frame titles include the protein name and UniProt accession when resolved.
`candidate_paths.html` and `py3dmol_path_view.html` display the same protein
label as a UniProt hyperlink when a valid accession exists. Metadata resolution
uses run metadata and caches before a short UniProt API request; offline or API
failure falls back to cached data or the accession alone.

PDB endpoint experiment outputs:

```text
outputs/pdb_endpoint_experiment/
  experiment_summary.json
  path_summary.json
  graph.png
  py3dmol_path_view.html
  pymol_path/
    view_path.pml
    00_A.pdb
    ...
```

## `experiment_summary.json`

Full metadata summary for an experiment run.

Useful fields:

- `fallback_used`: whether waypoint fallback was triggered.
- `n_nodes`: graph node count.
- `n_edges`: graph edge count.
- `path_node_ids`: selected shortest path.
- `path_intermediates`: number of non-endpoint path nodes.
- `route_family_costs`: fallback route costs when applicable.
- `candidate_paths`: candidate-report schema, file references, and exact/diverse
  counts when candidate reporting is enabled. The scan count is stored in
  `candidate_paths.json`.
- `protein_metadata`: resolved name, accession, display label, UniProt URL, and
  resolution source used by GIF and HTML writers.
- `generation_summaries`: Gaussian generation acceptance/rejection metadata.
- `input`: input mode and PDB matching metadata when applicable.

## `path_summary.json`

Compact shortest-path record from `PathResult.to_dict()`.

Includes:

- ordered node IDs
- total path cost
- per-edge scores and weights
- optional unnormalized probability diagnostics

## `graph.png`

Static NetworkX graph view.

Important interpretation:

- This is an abstract graph layout, not a protein structure.
- Edge `weight` is a cost.
- Path edges are dark/thick.
- Non-path edges may be filtered or faded by experiment graph display settings.

## `py3dmol_path_view.html`

Standalone browser view of the protein-like C-alpha structures on the selected path.

Important interpretation:

- This shows path conformations, not the NetworkX graph itself.
- Each overlaid structure is one path node.
- Current structures are C-alpha traces only.

## `pymol_path/`

PyMOL-compatible path view.

- `view_path.pml`: PyMOL script.
- `*.pdb`: C-alpha-only structures for A, B, and path nodes.

Open `view_path.pml` in PyMOL to inspect the same selected path shown in py3Dmol.

The categorized GIFs do not enable the heavier per-path PyMOL and py3Dmol
artifacts. Set `write_alternative_path_visuals` or the corresponding CLI flag
to emit those artifacts for reported alternatives; candidate JSON/HTML reports
and categorized GIFs are independent of that optional rendering.

## Git Hygiene

Generated output files should usually remain uncommitted.

The tracked `.gitkeep` keeps the folder present, and this README documents expected artifacts.
