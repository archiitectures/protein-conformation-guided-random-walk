# Scripts Walkthrough

This folder contains command-line entry points. Scripts are thin interfaces over the package code.

## `build_dataset.py`

Builds candidate experimental endpoint datasets from one mutually exclusive
input mode: `--manifest`, `--uniprot-ids`, `--uniprot-ids-file`, or `--pdb-ids`.
Use `--resume` for matching stage checkpoints and `--force` to recompute.

## `inspect_entry.py`

Prints endpoint metadata, sequence identity, aligned coverage, common C-alpha
count, RMSD, missing-residue fractions, ligand and partner differences,
mutations, warnings, rejection reasons, sequence selection, and artifact paths.

## `run_demo.py`

Purpose: quick end-to-end smoke demo.

Behavior:

1. Loads PDB endpoints if both `--endpoint-a` and `--endpoint-b` are supplied.
2. Otherwise creates small synthetic toy endpoints.
3. Generates Gaussian perturbation nodes from A and B.
4. Builds a weighted graph.
5. Runs Dijkstra plus bounded exact/diverse candidate-path search.
6. Writes:
   - `outputs/path_summary.json`
   - `outputs/candidate_paths.json`
   - `outputs/candidate_paths.html`
   - `outputs/graph.png`
   - `outputs/nodes/*.pdb`

Typical command:

```bash
uv run python scripts/run_demo.py
```

With PDB endpoints:

```bash
uv run python scripts/run_demo.py \
  --endpoint-a data/pdb_inputs/a.pdb \
  --endpoint-b data/pdb_inputs/b.pdb \
  --n-samples 20 \
  --scorer rmsd
```

This script is intentionally simple and still uses the older small demo pathway. For exploratory large tests, use `run_large_synthetic_experiment.py`.

## `run_large_synthetic_experiment.py`

Purpose: command-line interface for the larger experiment implemented in `guided_conformational_path.large_synthetic`.

Default command:

```bash
uv run python scripts/run_large_synthetic_experiment.py
```

High-level behavior:

1. Creates compact globular synthetic A/B endpoints.
2. Attempts recursive Gaussian sampling from both ends.
3. Falls back to route-constrained waypoint families if needed.
4. Builds graph and finds the A-to-B shortest path.
5. Writes NetworkX, morph GIF, PyMOL, and py3Dmol outputs.

Useful parameters:

```bash
--n-residues 200
--n-samples 80
--noise-scale 0.15
--recursive-max-iterations 6
--recursive-top-k 8
--recursive-samples-per-parent 4
--no-fallback
--fallback-waypoints 4
--fallback-samples-per-waypoint 0
--k-neighbors 0
--outputs-dir outputs/large_synthetic
--num-top-gifs-generated 6
--num-bottom-gifs-generated 1
--animation-frames-per-edge 20
--animation-fps 10
--animation-dpi 120
```

Graph display controls:

```bash
--graph-min-edge-weight 0.0
--graph-max-edge-weight 150.0
--graph-non-path-edge-alpha 0.125
```

Since edge `weight` is a cost, `--graph-max-edge-weight` is usually the useful filter for hiding high-cost non-path edges.

If `--no-fallback` is supplied, the script keeps the recursive Gaussian result
even when it misses the configured target intermediate-count window.

Visualization output controls:

```bash
--no-animation
--no-pymol
--no-py3dmol
```

The default animation set keeps `path_morph.gif` as the canonical best-path
animation and requests seven categorized files under `path_gifs/`: three exact
top paths, three diverse alternatives, and one deterministic non-top path.
`--num-top-gifs-generated 0` or `--num-bottom-gifs-generated 0` disables that
category; either option rejects negative values. `--no-animation` skips both
the canonical GIF and all categorized GIFs. If fewer qualifying paths exist,
the script writes only those available.

Selected `random_non_top` paths are also written to `candidate_paths.json` and
shown in a separate table in `candidate_paths.html`, with costs, deltas,
edge-overlap/diversity, intermediate overlap, per-edge diagnostics, and an HTML
graph thumbnail. They are comparison samples, not worst-path claims.

Outputs:

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

## `run_bioemu_precomputed_experiment.py`

Purpose: local macOS analysis of BioEmu samples generated elsewhere, usually in
Google Colab.

Default command:

```bash
uv run python scripts/run_bioemu_precomputed_experiment.py \
  --bioemu-bundle-dir data/bioemu_outputs/latest
```

High-level behavior:

1. Loads endpoint PDBs supplied by `--endpoint-a` and `--endpoint-b` (the
   checked-in demo fixtures are `data/pdb_inputs/a.pdb` and `b.pdb`).
2. Loads `manifest.json` plus generated PDB samples from the BioEmu bundle.
3. Treats BioEmu samples as neutral `external` graph nodes.
4. Builds the weighted graph, finds the A-to-B shortest path, and writes outputs.

Useful parameters:

```bash
--endpoint-a data/pdb_inputs/a.pdb
--endpoint-b data/pdb_inputs/b.pdb
--bioemu-bundle-dir data/bioemu_outputs/latest
--max-bioemu-samples 200
--outputs-dir outputs/bioemu_experiment
--num-top-gifs-generated 6
--num-bottom-gifs-generated 1
--no-pymol
--no-py3dmol
--no-animation
```

## `run_dataset_bioemu_experiment.py`

Purpose: automatic local graph analysis for BioEmu samples generated from a
dataset pair's selected sequence in Colab.

Typical command:

```bash
uv run python scripts/run_dataset_bioemu_experiment.py \
  --pair-dir data/dataset/first_five_proteins/P69441/pairs/4AKE_A__1AKE_A \
  --bioemu-bundle-dir data/bioemu_outputs/first_five_proteins/P69441/4AKE_A__1AKE_A/n_200 \
  --outputs-dir outputs/dataset_bioemu/P69441/4AKE_A__1AKE_A/n_200 \
  --k-neighbors 10
```

High-level behavior:

1. Loads the dataset pair's selected BioEmu sequence, aligned endpoints, and
   common-residue mapping.
2. Verifies that the downloaded Colab bundle contains that exact sequence and
   full-length C-alpha samples.
3. Selects the BioEmu coordinates corresponding to residues observed in both
   endpoints and aligns them into the endpoint frame.
4. Builds the graph, runs Dijkstra path search, and writes the standard graph,
   JSON, GIF, PyMOL, and py3Dmol outputs.

Useful parameters:

```bash
--max-bioemu-samples 200
--k-neighbors 10
--write-nodes
--num-top-gifs-generated 6
--num-bottom-gifs-generated 1
--no-animation
--no-pymol
--no-py3dmol
```

Use this script for endpoint pairs exported by the dataset generator. Use
`run_bioemu_precomputed_experiment.py` only for the older workflow with two
standalone endpoint PDB files that already match every BioEmu sample row.

Both analyzers are graph-only rerun commands when their input bundle already
exists: they reload normalized bundle nodes and regenerate graph search,
candidate reports, GIFs, PyMOL, and py3Dmol outputs without invoking BioEmu.
Choose a new `--outputs-dir` when preserving an earlier analysis matters.
Reusing an output directory is also safe for categorized GIFs: the writer
removes stale `top_exact_*`, `top_diverse_*`, and `random_non_top_*` GIFs that
it owns, but preserves unrelated files in `path_gifs/` and the rest of the
output tree.

Dataset-aware runs carry the protein name and UniProt accession from the pair
manifest. Standalone runs can infer a valid accession from BioEmu sequence
metadata. GIF titles show the resolved name/accession, while
`candidate_paths.html` and `py3dmol_path_view.html` hyperlink the label to the
UniProt entry. Resolution uses local metadata and caches first, attempts the
UniProt API only when needed, and remains usable offline with an accession-only
fallback.

## `colab_transfer.py`

Purpose: transfer the current worktree and a selected exported dataset through
the VS Code Colab extension without Google Drive.

Build the upload archive:

```bash
uv run python scripts/colab_transfer.py package-workspace
```

This writes `outputs/colab_transfers/guided_conformation_workspace.zip` with
only runtime source, project metadata, and the selected dataset. It excludes
Git history, virtual environments, caches, tests, notebooks, and prior outputs.
Use `--dataset-root data/dataset/non_ideal_transitions_n3` for the hard-target
experiment. Result archives carry their dataset label and import into isolated
BioEmu and graph-output roots.

Import one downloaded pair result:

```bash
uv run python scripts/colab_transfer.py import-result \
  outputs/colab_transfers/downloads/P69441__4AKE_A__1AKE_A__n_200.zip
```

The importer validates archive paths, schema, identifiers, sample counts, and
required bundle/graph files before writing canonical local destinations. It
refuses to overwrite an existing run unless `--force` is supplied.

## Parallel Project Scripts

The scripts under `parallel_bioemu_tools/scripts/` belong to the parallel
subproject in this same worktree. Its dependencies are isolated by its own
project metadata and must be run with that project selected:

```bash
UV_CACHE_DIR=.uv-cache uv --project parallel_bioemu_tools run \
  python parallel_bioemu_tools/scripts/run_worker.py --help
```

See `parallel_bioemu_tools/README.md` for the complete prepare, worker,
supervisor, watchdog, collector, and VS Code Colab commands.

## Design Rule For Scripts

Do not put new scientific or graph logic directly in scripts. Add reusable logic to `guided_conformational_path/`, then expose it with a script.
