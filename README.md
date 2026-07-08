# Protein Conformation Guided-Random Walk

This repository contains an initial research prototype for building candidate
conformational transition graphs between two endpoint protein structures.

> This prototype constructs candidate conformational transition graphs between two endpoint protein structures. Generated conformations are treated as candidate nodes, and edge costs are heuristic transition-plausibility scores. The resulting shortest path is a proposed structural path, not a validated physical transition pathway.

## What It Does

- Represents protein conformations as C-alpha coordinate nodes.
- Generates candidate nodes from endpoint structures using a dummy perturbation generator.
- Scores edges between conformations with RMSD or distance-histogram costs.
- Builds a weighted NetworkX graph over endpoints and generated conformations.
- Finds a lowest-cost path between endpoint A and endpoint B.
- Saves a JSON path summary, a graph PNG, and generated PDB-like node files.

## What It Does Not Claim

- It does not prove kinetic connectivity between generated conformations.
- It does not estimate MSM transition probabilities, stationary distributions, MFPTs, or free energies.
- It does not claim the shortest graph path is the true physical transition pathway.
- It does not run BioEmu, EigenFold, or another protein ensemble model yet.

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

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

The first version uses standard CPU-friendly dependencies only: NumPy, SciPy,
NetworkX, Matplotlib, and pytest.

## Run The Demo

Run with synthetic toy endpoints:

```bash
python scripts/run_demo.py
```

Run with PDB endpoints:

```bash
python scripts/run_demo.py \
  --endpoint-a data/examples/A.pdb \
  --endpoint-b data/examples/B.pdb \
  --n-samples 20 \
  --scorer rmsd
```

Outputs are written to:

- `outputs/path_summary.json`
- `outputs/graph.png`
- `outputs/nodes/`

## Run Tests

```bash
pytest
```

## Plugging In BioEmu Or EigenFold Later

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
