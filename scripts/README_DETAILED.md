# Scripts Walkthrough

This folder contains command-line entry points. Scripts are thin interfaces over the package code.

## `run_demo.py`

Purpose: quick end-to-end smoke demo.

Behavior:

1. Loads PDB endpoints if both `--endpoint-a` and `--endpoint-b` are supplied.
2. Otherwise creates small synthetic toy endpoints.
3. Generates Gaussian perturbation nodes from A and B.
4. Builds a weighted graph.
5. Runs shortest-path search.
6. Writes:
   - `outputs/path_summary.json`
   - `outputs/graph.png`
   - `outputs/nodes/*.pdb`

Typical command:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/run_demo.py
```

With PDB endpoints:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/run_demo.py \
  --endpoint-a data/examples/A.pdb \
  --endpoint-b data/examples/B.pdb \
  --n-samples 20 \
  --scorer rmsd
```

This script is intentionally simple and still uses the older small demo pathway. For exploratory large tests, use `run_large_synthetic_experiment.py`.

## `run_large_synthetic_experiment.py`

Purpose: command-line interface for the larger experiment implemented in `guided_conformational_path.large_synthetic`.

Default command:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/run_large_synthetic_experiment.py
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

Outputs:

```text
outputs/large_synthetic/
  experiment_summary.json
  path_summary.json
  graph.png
  path_morph.gif
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
UV_CACHE_DIR=.uv-cache uv run python scripts/run_bioemu_precomputed_experiment.py \
  --bioemu-bundle-dir data/bioemu_outputs/latest
```

High-level behavior:

1. Loads endpoint PDBs from `data/pdb_inputs/A.pdb` and `data/pdb_inputs/B.pdb`.
2. Loads `manifest.json` plus generated PDB samples from the BioEmu bundle.
3. Treats BioEmu samples as neutral `external` graph nodes.
4. Builds the weighted graph, finds the A-to-B shortest path, and writes outputs.

Useful parameters:

```bash
--endpoint-a data/pdb_inputs/A.pdb
--endpoint-b data/pdb_inputs/B.pdb
--bioemu-bundle-dir data/bioemu_outputs/latest
--max-bioemu-samples 200
--outputs-dir outputs/bioemu_experiment
--no-pymol
--no-py3dmol
--no-animation
```

## Design Rule For Scripts

Do not put new scientific or graph logic directly in scripts. Add reusable logic to `guided_conformational_path/`, then expose it with a script.
