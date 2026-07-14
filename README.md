# Protein Conformation Guided-Random Walk

This repository contains an initial research prototype for building candidate
conformational transition graphs between two endpoint protein structures.

> This prototype constructs candidate conformational transition graphs between two endpoint protein structures. Generated conformations are treated as candidate nodes, and edge costs are heuristic transition-plausibility scores. The resulting shortest path is a proposed structural path, not a validated physical transition pathway.

## What It Does

- Represents protein conformations as C-alpha coordinate nodes.
- Generates candidate nodes recursively from endpoint-seeded frontier structures
  using a dummy perturbation generator.
- Scores edges between conformations with RMSD or distance-histogram costs.
- Builds a weighted NetworkX graph over endpoints and generated conformations.
- Finds a lowest-cost path between endpoint A and endpoint B.
- Saves a JSON path summary, a graph PNG, a best-path morph GIF, and generated
  PDB-like node files.

## What It Does Not Claim

- It does not prove kinetic connectivity between generated conformations.
- It does not estimate MSM transition probabilities, stationary distributions, MFPTs, or free energies.
- It does not claim the shortest graph path is the true physical transition pathway.
- BioEmu support is limited to a Colab-facing sampling runner that writes
  generated structures as portable candidate-node bundles.

Generated plausible structures are candidate graph nodes only. Equilibrium-like
samples or model-generated conformations do not automatically determine
transition dynamics.

## Current Abstractions

- `ConformationNode` is the central data object. It stores a unique node ID,
  source label, optional metadata, optional source path, and C-alpha coordinates
  with shape `(L, 3)`.
- `NodeGenerator` defines how candidate nodes are proposed from an endpoint.
  The dummy generator adds Gaussian noise to C-alpha coordinates and rejects
  samples that fail lightweight validity checks.
- `EdgeScorer` defines graph edge costs. `RMSDScorer` uses Kabsch-aligned
  C-alpha RMSD. `DistanceHistogramScorer` compares global pairwise C-alpha
  distance histograms.
- `validity.py` provides coarse C-alpha checks for finite coordinates,
  consecutive C-alpha bond-distance ranges, and non-neighbor C-alpha
  clash-like overlaps.
- `build_conformation_graph` creates a NetworkX weighted graph. Edge weights are
  scorer costs plus an additive validity penalty.
- `find_shortest_path` uses Dijkstra shortest path over graph edge weights. Its
  optional `beta` parameter only reports unnormalized `exp(-beta * weight)`
  diagnostics; it does not change the path objective.

The clash filter is intentionally C-alpha-only because the current data model
does not store full atom coordinates. It helps reject severe coordinate overlaps
from Gaussian perturbations, but it is not a full atom-atom steric validation.
Full clash detection will require full-atom structures or atom reconstruction in
a later version.

## Setup

Using `uv`:

```bash
UV_CACHE_DIR=.uv-cache uv sync --extra test --extra notebook
```

Then run commands through the managed environment:

```bash
UV_CACHE_DIR=.uv-cache uv run pytest
UV_CACHE_DIR=.uv-cache uv run python scripts/run_demo.py
UV_CACHE_DIR=.uv-cache uv run python scripts/run_large_synthetic_experiment.py
```

`UV_CACHE_DIR=.uv-cache` keeps uv's package cache inside the project directory,
which is useful in restricted environments where the home cache is not writable.

The reusable model/experiment definitions for the larger synthetic test live in
`guided_conformational_path/large_synthetic.py`. The CLI script and notebook are
thin interfaces over that module.

To use the notebook interface:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m ipykernel install --user --name guided-conformational-path
```

Then open `notebooks/large_synthetic_experiment.ipynb` in Jupyter or VS Code and
select the `guided-conformational-path` kernel.

For PDB endpoint experiments, place files here:

```text
data/pdb_inputs/A.pdb
data/pdb_inputs/B.pdb
```

The notebook has a dedicated PDB-endpoint section that loads those files and
runs the same graph/path/3D-view workflow. The current PDB parser uses C-alpha
coordinates only.

PDB endpoint matching supports `matching_method="exact_then_geometric"`. The
loader first tries exact matching by chain ID, residue number, and insertion
code. If too few residue IDs overlap, it falls back to a Kabsch-aligned mutual
nearest-neighbor C-alpha match. This fallback is useful for exploratory tests
with very similar backbones, but it is heuristic and should not be treated as a
validated sequence or structural alignment.

Using standard `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test,notebook]"
```

The first version uses standard CPU-friendly dependencies only: NumPy, SciPy,
NetworkX, Matplotlib, Pillow, and pytest.

## Run The Demo

Run with synthetic toy endpoints:

```bash
python scripts/run_demo.py
```

Run with PDB endpoints:

```bash
python scripts/run_demo.py \
  --endpoint-a data/pdb_inputs/A.pdb \
  --endpoint-b data/pdb_inputs/B.pdb \
  --n-samples 20 \
  --scorer rmsd
```

Outputs are written to:

- `outputs/path_summary.json`
- `outputs/graph.png`
- `outputs/nodes/`

## Run The Larger Synthetic Experiment

For a more structured graph-behavior test, run:

```bash
python scripts/run_large_synthetic_experiment.py
```

By default this creates two 200-C-alpha synthetic endpoints, runs recursive
Gaussian sampling from both ends, and falls back to route-constrained waypoint
families if the recursive graph does not produce a 3-4 intermediate path.
Change the size with:

```bash
python scripts/run_large_synthetic_experiment.py --n-residues 500
```

Use `--no-fallback` to inspect the recursive Gaussian result directly even when
it misses the configured 3-4-intermediate target window.

The primary phase still uses `DummyPerturbationGenerator`, but it no longer
samples only from A and B. It starts from both endpoints, generates noisy
children, keeps frontier parents from the closest cross-side A/B candidate
pairs, and repeats up to `recursive_max_iterations`. `n_samples` is the total
accepted recursive-node cap across both sides, while
`recursive_top_k` and `recursive_samples_per_parent` control frontier breadth
and local branching. The fallback creates several candidate waypoint routes and
reports route-family costs so you can see which path wins under the
experiment's squared-RMSD transition score.

For large NetworkX graph views, the experiment config supports sparse display
controls:

```python
graph_min_edge_weight=None
graph_max_edge_weight=None
graph_non_path_edge_alpha=0.125
```

Because edge `weight` is a cost, `graph_max_edge_weight` is usually the most
useful filter: it hides high-cost non-path edges. Path edges are always drawn.

The larger experiment also writes a PyMOL-ready view under
`outputs/large_synthetic/pymol_path/`. Open `view_path.pml` in PyMOL to load A,
B, and the nodes on the selected A-to-B path with distinct colors. If PyMOL is
available locally, the script can also render `path_nodes.png` from inside
PyMOL. Use `--no-pymol` to skip these files.

It also writes `outputs/large_synthetic/py3dmol_path_view.html`, a standalone
3Dmol.js/py3Dmol-style browser view that overlays A, B, and the selected path
nodes. In that view, each colored C-alpha trace is one conformation from the
selected graph path. Larger labeled spheres mark graph-node centroids, and black
cylinders connect consecutive nodes on the best path. In notebooks, call
`display_py3dmol_path(run)` from
`guided_conformational_path.large_synthetic` to show the same structure path
inline next to the NetworkX graph outputs. Use `--no-py3dmol` to skip the HTML
view.

The larger experiment also writes `outputs/large_synthetic/path_morph.gif`, a
Matplotlib-rendered morph animation of the selected best-path C-alpha
backbone. Use `--no-animation` to skip that file, or tune the GIF with
`--animation-frames-per-edge`, `--animation-fps`, and `--animation-dpi`.

## Run Tests

```bash
pytest
```

## BioEmu Colab Bundle Runner

`guided_conformational_path.bioemu_runner.run_bioemu_from_fasta` is a thin
adapter for running BioEmu in environments such as Google Colab. It reads a FASTA
file, requests BioEmu samples, and writes a portable bundle:

- `manifest.json`
- `nodes/*.pdb`
- `raw_bioemu/`

The runner keeps BioEmu imports lazy, so installing this package does not require
BioEmu. In Colab, open `notebooks/run_bioemu_colab.ipynb`, select a GPU runtime,
install `bioemu[cuda]`, mount Google Drive, set the FASTA/output paths, and call
the runner.

Because BioEmu APIs can vary across releases and deployments, the runner exposes
`sampler=` as the adapter point. Pass a callable that accepts some subset of
`fasta_path`, `sequence`, `output_dir`, `n_samples`, `num_samples`, and `seed`.
The callable may return PDB paths, PDB text strings, coordinate arrays with shape
`(L, 3)` or `(N, L, 3)`, or write `.pdb` files into the provided output
directory.

BioEmu-generated structures are candidate graph nodes only. They do not by
themselves establish transition dynamics, kinetic connectivity, or free-energy
estimates.

To analyze a completed or downloaded BioEmu bundle locally, keep endpoint files
at:

```text
data/pdb_inputs/A.pdb
data/pdb_inputs/B.pdb
```

Then run:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/run_bioemu_precomputed_experiment.py \
  --bioemu-bundle-dir data/bioemu_outputs/latest \
  --outputs-dir outputs/bioemu_experiment
```

The bundle must contain `manifest.json` and normalized generated PDB files
referenced by that manifest. The local analyzer treats BioEmu structures as
neutral `external` graph nodes, then writes the same graph, path, py3Dmol,
PyMOL, and morph-GIF outputs as the synthetic/PDB workflows.

## Plugging In New Generators Later

Future ensemble generators should implement the `NodeGenerator` interface:

```python
class NodeGenerator:
    def generate_from_endpoint(self, endpoint, n_samples, **kwargs):
        ...
```

The package already includes placeholder `BioEmuNodeGenerator` and
`EigenFoldNodeGenerator` classes. A future implementation should call the
external model, convert generated structures into C-alpha coordinate arrays, and
return `ConformationNode` objects. The graph, scoring, validity, visualization,
and path-search layers should not need to know which generator produced the
nodes.

The dummy Gaussian generator may return fewer than the requested number of nodes
if validity filtering rejects too many proposed samples within its attempt
budget. The generator records accepted/rejected counts in
`last_generation_summary` and in accepted node metadata.

## Checkpoint And Resume

Checkpoint/resume is intentionally only a placeholder in this MVP. There is no
long-running training or inference process yet. The demo exposes `--resume` to
reserve the interface for future expansion.
