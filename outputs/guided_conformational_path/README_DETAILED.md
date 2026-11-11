# Package Walkthrough

This folder contains the reusable Python package code. Scripts and notebooks should import from here rather than duplicating experiment logic.

## Core Data Model

`nodes.py`

- Defines `ConformationNode`, the central object passed through the package.
- Each node has:
  - `node_id`: unique graph identifier.
  - `source`: one of `A_endpoint`, `B_endpoint`, `A_generated`, `B_generated`, or `external`.
  - `coords`: NumPy array of C-alpha coordinates with shape `(L, 3)`.
  - `metadata`: provenance and debugging details.
  - `file_path`: optional source PDB path.
- Current limitation: the project stores C-alpha traces only, not full atomistic structures.

## PDB I/O

`io.py`

- `read_pdb_ca(path)` returns only a coordinate array.
- `read_pdb_ca_records(path)` returns `CARecord` objects with residue metadata:
  - chain ID
  - residue number
  - insertion code
  - residue name
  - coordinate
- `write_pdb_ca(path, coords)` writes minimal PDB-like C-alpha files.

Important caveat: this is a simple fixed-column PDB parser/writer, not a full PDB/mmCIF implementation.

## Node Generation

`generators.py`

- `NodeGenerator` is the interface for anything that proposes conformational nodes.
- `DummyPerturbationGenerator` adds Gaussian noise to endpoint C-alpha coordinates.
- The dummy generator can reject generated nodes that fail validity checks.
- `PrecomputedNodeGenerator` loads PDB files from a directory.
- `BioEmuNodeGenerator` and `EigenFoldNodeGenerator` are placeholders.

The dummy generator is for pipeline testing. It is not a physical protein sampler.

## BioEmu Colab Runner

`bioemu_runner.py`

- Provides `run_bioemu_from_fasta(...)` for Colab-facing BioEmu sampling.
- Keeps BioEmu imports lazy inside the runner function, so BioEmu is not a
  package dependency.
- Reads the first FASTA record, requests `n_samples` samples, and writes a
  portable bundle with:
  - `manifest.json`
  - `nodes/*.pdb`
  - `raw_bioemu/`
- Exposes `sampler=` as the adapter point for BioEmu API differences.
- Normalizes supported sampler outputs: PDB paths, PDB text strings, coordinate
  arrays with shape `(L, 3)` or `(N, L, 3)`, or `.pdb` files written under the
  raw output directory.
- Recognizes the official `bioemu.sample.main` Python entry point.
- Expands official `topology.pdb` plus `samples.xtc` output into one C-alpha PDB
  node per physical trajectory frame. `mdtraj` is imported lazily from the
  BioEmu environment, and nanometer coordinates are converted to Angstrom.
- Excludes trajectories marked as unphysical.

BioEmu samples are treated as external candidate nodes. They do not establish
kinetic connectivity or physical transition dynamics by themselves.

The bundle records both `n_samples_requested` and `n_samples_written`. The
requested value is a target passed to BioEmu; the written value is the number
of normalized, retained PDB nodes after trajectory conversion and filtering.
Downstream graph analysis operates on retained nodes only and must not infer
that a requested count was physically produced.

`bioemu_io.py`

- Defines the portable `bioemu_bundle_v1` manifest shape.
- Loads `manifest.json` plus normalized BioEmu PDB samples.
- Converts samples into `ConformationNode(source="external")`.
- Validates optional residue-count compatibility with endpoint A/B.

`bioemu_experiment.py`

- Runs local graph/path analysis for a precomputed BioEmu bundle.
- Uses endpoint PDB files plus BioEmu bundle samples.
- Reuses the same graph, path-search, and output writers as the synthetic/PDB
  workflow.

`dataset_bioemu_adapter.py`

- Bridges an exported `protein_conformer_dataset` endpoint pair to a full-sequence
  Colab BioEmu bundle.
- `load_dataset_pair_selection(...)` loads the pair manifest, aligned endpoint
  arrays, selected sequence, and common-residue mapping.
- `load_adapted_bioemu_nodes(...)` requires exact sequence identity, verifies
  full sample length, selects common endpoint residues, and rigidly aligns each
  sample into the endpoint coordinate frame.
- Stores one-based sequence positions on every adapted node so missing-residue
  gaps remain explicit during validity checks.
- Raises `DatasetBioEmuAdapterError` rather than guessing when mapping or
  sequence provenance is ambiguous.

`dataset_bioemu_experiment.py`

- Defines `DatasetBioEmuConfig` for the one-command Colab-to-local handoff.
- `run_dataset_bioemu_experiment(...)` invokes the adapter, graph construction,
  Dijkstra path search, validity summaries, and standard experiment metadata.
- Does not modify the downloaded BioEmu bundle. `--write-nodes` can be used by
  the output layer when persisted adapted coordinates are useful.

`colab_transfer.py`

- Defines versioned workspace and pair-result ZIP schemas for the no-Drive VS
  Code Colab workflow.
- `create_workspace_archive(...)` packages runtime source plus one or more
  exported endpoint pairs under the supplied dataset label while excluding local
  environments and caches.
- `create_result_archive(...)` combines one complete BioEmu bundle and graph
  output directory with provenance and retained-sample metadata.
- `import_result_archive(...)` rejects unsafe members, invalid schemas,
  mismatched identifiers, incomplete outputs, and implicit overwrite before
  restoring canonical local paths.

## Scoring

`scoring.py`

- `EdgeScorer` defines the edge-cost interface.
- `RMSDScorer` computes Kabsch-aligned C-alpha RMSD.
- `DistanceHistogramScorer` compares pairwise C-alpha distance histograms.
- `kabsch_rmsd(...)` assumes both coordinate arrays already represent matching residue order.

Lower scores mean cheaper/more plausible graph edges.

## Validity

`validity.py`

- `check_node_validity(node)` returns `ValidityResult`.
- Checks include:
  - coordinate shape
  - finite values
  - consecutive C-alpha bond-distance bounds
  - coarse non-neighbor C-alpha clash-like distances
- If `node.metadata["residue_positions"]` is present, only true consecutive
  sequence positions are treated as bonded neighbors. This prevents unresolved
  endpoint residues from producing false bond penalties or hiding clashes.

The clash check is C-alpha-only. It does not detect full atom-atom clashes.

## Graph Construction

`graph.py`

- `build_conformation_graph(...)` creates a NetworkX weighted graph.
- Nodes are `ConformationNode` instances.
- Edges store:
  - `score`: raw scorer cost
  - `weight`: scorer cost plus validity penalty
- `k_neighbors=None` creates a fully connected graph.
- `k_neighbors=N` keeps each node's lowest-cost incident edges.

Since edge `weight` is a cost, smaller is better.

## Path Search

`path_search.py`

- `find_shortest_path(...)` runs NetworkX Dijkstra search.
- `find_top_k_paths(...)` performs one bounded Yen simple-path scan and returns
  `CandidatePathCollection`: the Dijkstra path plus exact and edge-diverse
  alternatives. `find_candidate_paths(...)` exposes either candidate subset.
- Candidate metrics include structural-cost deltas, edge Jaccard/diversity,
  intermediate-node Jaccard, global cost rank, and per-edge diagnostics.
- `candidate_report.py` writes versioned JSON and sortable, escaped HTML with a
  deterministic shared NetworkX layout. When a valid UniProt accession is
  available, its protein heading links to the canonical UniProt entry. This
  comparison does not imply kinetic connectivity or physical transition
  probabilities.
- Selected `random_non_top` comparison paths are included in the JSON and in a
  separate HTML table with cost/delta, edge overlap/diversity, intermediate
  overlap, per-edge diagnostics, and the shared-layout graph thumbnail.
- Returns `PathResult` with:
  - ordered node IDs
  - total cost
  - per-edge records
  - metadata
- Optional `beta` only adds an unnormalized `exp(-beta * weight)` diagnostic. It does not change the path objective.

## Visualization

`visualization.py`

- `plot_graph(...)` writes a static NetworkX graph PNG.
- Endpoint nodes are larger.
- Selected path edges are drawn dark and thick.
- Non-path edges can be filtered and faded:
  - `min_edge_weight`
  - `max_edge_weight`
  - `non_path_edge_alpha`
  - `path_edge_alpha`

For this project, `max_edge_weight` is usually the useful sparse-view control because edge weights are costs.

Layout note: the stored graph edge `weight` is a cost, but NetworkX spring
layouts interpret larger weights as stronger attraction. The renderer therefore
uses an internal inverse-cost `layout_weight` for layout only; it does not alter
the graph's stored edge costs.

## Large Synthetic And PDB Endpoint Experiment

`large_synthetic.py`

This is the main experiment module used by the CLI and notebook.

Major config objects:

- `LargeSyntheticConfig`: generated globular A/B endpoints.
- `PDBEndpointConfig`: PDB-provided A/B endpoints.
- Both configs support `write_animation`, `animation_frames_per_edge`,
  `animation_fps`, `animation_dpi`, `num_top_gifs_generated`, and
  `num_bottom_gifs_generated`.

Main run functions:

- `run_large_synthetic_experiment(config)`
- `run_pdb_endpoint_experiment(config)`
- `write_run_outputs(run, config)`

Notable controls:

- `allow_fallback=False` keeps the recursive Gaussian graph/path even when it
  misses the configured target intermediate-count range.
- `progress_callback=` lets notebooks report recursive-iteration progress.

The experiment tries:

1. Recursive Gaussian node generation using `DummyPerturbationGenerator`.
2. Closest cross-side A/B candidate pairs are used to choose the next frontier.
3. If that does not produce the target path shape, fallback route families:
   - `low_energy`
   - `medium_detour`
   - `high_detour`

PDB endpoint matching:

- First tries exact matching by `(chain_id, residue_number, insertion_code)`.
- If exact overlap is insufficient, falls back to Kabsch-aligned mutual nearest-neighbor C-alpha matching.
- Geometric matching is heuristic and should be treated as exploratory only.

3D outputs:

- `path_morph_frames(...)` builds interpolated best-path backbone frames.
- `write_path_morph_gif(...)` writes a Matplotlib/Pillow GIF for the selected
  path. `path_morph.gif` remains the canonical best-path filename.
- Categorized animation output goes under `path_gifs/`. Defaults request three
  exact GIFs, three diverse GIFs, and one seed-deterministic non-top GIF, using
  `top_exact_<category-rank>_global_<cost-rank>.gif`,
  `top_diverse_<category-rank>_global_<cost-rank>.gif`, and
  `random_non_top_<category-rank>.gif` names. Zero disables a category and
  negative counts raise `ValueError` before rendering.
- Before a rerun writes categorized animations, it removes only stale GIFs
  matching those feature-owned naming patterns. Unrelated files in
  `path_gifs/` and elsewhere under the output directory are preserved.
- `write_pymol_path_view(...)` writes path-node PDBs and `view_path.pml`.
- `write_py3dmol_path_html(...)` writes standalone 3Dmol.js HTML.
- `display_py3dmol_path(...)` returns an inline notebook viewer.

`protein_metadata.py` resolves visualization labels from existing run metadata,
the temporary JSON cache, repository UniProt response caches, or the
UniProt API in that order. Network and cache failures are non-fatal: a valid
accession remains the offline display fallback. GIF titles include the resolved
name/accession, and candidate-report and py3Dmol HTML headings hyperlink valid
accessions to UniProt.

`run_dataset_bioemu_experiment(...)` and
`run_bioemu_precomputed_experiment(...)` only consume existing normalized
BioEmu bundles. Calling them again rebuilds the graph, path candidates, reports,
and visuals without importing or running the BioEmu sampler.

## Adding New Functionality

Prefer this layering:

1. Add reusable logic in this package.
2. Add a thin CLI wrapper under `scripts/`.
3. Add notebook cells that call package functions.
4. Add tests under `tests/`.

Avoid putting core experiment logic directly in notebooks.
