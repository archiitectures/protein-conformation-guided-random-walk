# Protein Conformation Guided Random Walk

This repository is maintained as one worktree inside the `aa_overall_proj_1`
mother folder. Commands below assume the repository root as the working
directory. The root package uses the root `uv` environment; the optional
`parallel_bioemu_tools/` subproject has its own `pyproject.toml` and lockfile
but remains in this same worktree.

This repository is a research prototype for building candidate conformational
graphs between two experimentally observed protein endpoint structures. The
current end-to-end workflow is:

1. Discover or declare endpoint pairs using UniProt and RCSB/PDB metadata.
2. Parse the endpoint chains, align their amino-acid sequences, and map
   residues to common C-alpha coordinate positions.
3. Run BioEmu in a CUDA Colab runtime, or provide a compatible precomputed
   bundle.
4. Validate and locally adapt the full-sequence BioEmu samples to the common
   endpoint residue set.
5. Build a weighted graph, preserve the Dijkstra best path, enumerate nearby
   simple paths, and write graph, JSON, HTML, morph-animation, PyMOL, and
   py3Dmol outputs.

The result is a proposed structural route through candidate nodes. It is not a
validated kinetic pathway. Candidate paths are not validated kinetics, and the
repository does not estimate transition probabilities, MSM quantities,
free-energy surfaces, or mechanistic truth.

## Quick Start

The project uses `uv` and keeps its cache in the repository through `uv.toml`:

```bash
uv sync --extra test --extra notebook
uv run pytest
```

Run the CPU-only smoke demo:

```bash
uv run python scripts/run_demo.py
```

Run the larger synthetic graph experiment:

```bash
uv run python scripts/run_large_synthetic_experiment.py
```

For the notebook workflow, register the synced environment and select the
project kernel in VS Code or Jupyter:

```bash
uv run python -m ipykernel install --user \
  --name guided-conformational-path \
  --display-name "guided-conformational-path (.venv)"
```

No BioEmu or heavyweight ML framework is installed in the local environment.
BioEmu sampling is performed in CUDA Colab; graph analysis and collection run
locally.

## Dataset To Graph Workflow

Build a standardized endpoint dataset from the tracked explicit manifest:

```bash
uv run python scripts/build_dataset.py \
  --manifest data/input/endpoints.csv \
  --output-dir data/dataset
```

Or discover endpoint candidates from UniProt/PDB endpoint pairs:

```bash
uv run python scripts/build_dataset.py \
  --uniprot-ids P69441 \
  --output-dir data/dataset \
  --top-k-pairs 3 --resume
```

The dataset pipeline fetches canonical UniProt metadata, queries exact
UniProt-to-RCSB polymer mappings, downloads cached mmCIF files, parses one
protein chain per structure, aligns endpoint sequences, maps construct and
canonical residue positions, evaluates quality/context warnings, ranks pairs,
and exports manifests and review tables. It does not use author residue
numbering as the coordinate correspondence when sequence alignment is
available. See [the dataset package README](protein_conformer_dataset/README_DETAILED.md)
for input modes, QC, checkpoints, and export contents.

Inspect an exported pair:

```bash
uv run python scripts/inspect_entry.py \
  --dataset-dir data/dataset \
  --protein-id adenylate_kinase \
  --pair-id 4AKE_A__1AKE_A
```

The five-pair manifest used by the Colab workflow is tracked at
`data/input/first_five_proteins.csv`; its generated dataset under
`data/dataset/first_five_proteins/` is local and ignored. The five pairs are
currently `manual_review` rather than `accepted`, because the exports retain
ligand, mutation, construct, source, or length/context caveats. Generated
cache, dataset, BioEmu, and analysis content should not be committed.

## BioEmu Handoff

For a dataset pair, use its `sequence/bioemu_selected.fasta` in Colab. The
portable `bioemu_bundle_v1` result contains `manifest.json`, normalized
`nodes/*.pdb`, and `raw_bioemu/`. The manifest records the selected sequence,
requested sample count, retained sample count, and node paths. A request for
200 samples is a target: BioEmu may retain fewer physical trajectory frames,
and outputs labeled unphysical are not converted into graph nodes.

The supported no-Drive VS Code Colab flow packages source and the locally
generated five-pair dataset:

```bash
uv run python scripts/colab_transfer.py package-workspace
```

Open `notebooks/run_bioemu_colab.ipynb`, choose an NVIDIA `L4` or `T4` Colab
server through the VS Code Colab extension, and upload
`outputs/colab_transfers/guided_conformation_workspace.zip`. The notebook runs
one smoke task and then one production task per pair. Download each result ZIP
before starting the next task because `/content` is ephemeral. Import a result
with the schema/path validator:

```bash
uv run python scripts/colab_transfer.py import-result \
  outputs/colab_transfers/downloads/P69441__4AKE_A__1AKE_A__n_200.zip
```

Imported bundles are placed at
`data/bioemu_outputs/first_five_proteins/<protein>/<pair>/n_<requested>/`; graph
outputs are placed at `outputs/dataset_bioemu/<protein>/<pair>/n_<requested>/`.
These are ignored generated paths, not checked-in results. At this checkout,
local production results are partial; do not assume the P00889 or P35557
archives are present until they have been downloaded and imported.
For local analysis, use `scripts/run_dataset_bioemu_experiment.py` and the
dataset pair directory rather than manually copying endpoint PDBs.
That command is also the graph-only rerun path: point it at an existing BioEmu
bundle and a new output directory to rebuild graph search, reports, and visuals
without sampling with BioEmu again. Use
`scripts/run_bioemu_precomputed_experiment.py` for the analogous standalone-PDB
bundle workflow.

The independent parallel workflow is documented in
[parallel_bioemu_tools/README.md](parallel_bioemu_tools/README.md). It shards
requests into deterministic chunks, stores verified bundles and completion
markers in Hugging Face, watches heartbeats, queues recovery for stale workers,
collects durable results locally, and releases a Colab runtime only after the
required remote or local durability checks succeed.

## Graph And Visualization

`guided_conformational_path` represents each endpoint or generated structure as
a C-alpha `ConformationNode`. RMSD and distance-histogram scorers provide edge
costs; validity checks add a coarse C-alpha penalty; NetworkX stores the graph;
and Dijkstra returns the canonical minimum-cost path. Yen's algorithm then
enumerates simple paths in cost order. By default the report contains four
exact paths including Dijkstra plus up to four non-best alternatives with at
least 20% edge-set diversity from Dijkstra. `k_neighbors=0` means fully
connected, while a positive value sparsifies the graph.

Typical outputs include:

- `experiment_summary.json` and `path_summary.json` for provenance and path data.
- `candidate_paths.json` and sortable `candidate_paths.html` for exact,
  diversity-filtered, and selected `random_non_top` comparison paths. Reports
  include costs, deltas, overlap/diversity metrics, and edge diagnostics; the
  HTML adds shared-layout graph thumbnails.
- `graph.png` for the weighted NetworkX view.
- `path_morph.gif` for the canonical best-path C-alpha animation.
- `path_gifs/` for categorized path animations. Defaults write six top GIFs,
  split as three exact and three diverse paths, plus one deterministic non-top
  path selected outside the reported top-path union.
- `pymol_path/view_path.pml` and path-node PDB files.
- `py3dmol_path_view.html` for a standalone browser view.

Use `--path-top-k`, `--min-path-edge-diversity`, and
`--path-candidate-scan-limit` to tune candidate search. Use
`--num-top-gifs-generated` and `--num-bottom-gifs-generated` to control the
categorized GIF counts; zero disables that category and negative values are
invalid. `--no-animation` disables both the canonical and categorized GIFs.
PyMOL and py3Dmol remain best-path-only unless
`--write-alternative-path-visuals` is set.
On rerun, the output writer removes stale feature-owned categorized GIF names
before writing the new set, while preserving unrelated files in `path_gifs/`
and elsewhere in the output directory.

When a valid UniProt accession is available, GIF titles include the resolved
protein name and accession, and the candidate-path and py3Dmol HTML pages link
that label to UniProt. Name resolution prefers run metadata and cached values,
can query UniProt when online, and falls back safely to an accession-only label
when offline. These costs and rankings are heuristics, not calibrated energies
or path probabilities.

The C-alpha validity and clash checks are coarse and are not full-atom
structural validation. A shortest graph path is a heuristic candidate route,
not evidence of kinetic connectivity.

## Directory Responsibilities

- `protein_conformer_dataset/`: UniProt/RCSB discovery, parsing, alignment,
  residue mapping, QC, checkpointing, and dataset export.
- `guided_conformational_path/`: node model, BioEmu normalization and adapter,
  graph construction, path search, validity, and visualization.
- `scripts/`: thin CLI entry points for dataset creation, transfer, experiments,
  inspection, and demos.
- `notebooks/`: UI layers for synthetic/PDB experiments, Colab execution, and
  local post-download verification.
- `parallel_bioemu_tools/`: same-worktree shard workers, Hugging Face durability,
  heartbeat/watchdog/recovery, VS Code Colab batching, and local collection.
- `tests/`: offline pytest coverage for dataset, BioEmu handoff, notebooks,
  graph behavior, and parallel orchestration.
- `data/`: small inputs and generated local caches/datasets; see
  [data/README_DETAILED.md](data/README_DETAILED.md).
- `outputs/`: generated experiment, transfer, imported bundle, and graph files.

Read the folder-specific detailed READMEs before changing a subsystem:

- [package details](guided_conformational_path/README_DETAILED.md)
- [dataset details](protein_conformer_dataset/README_DETAILED.md)
- [script details](scripts/README_DETAILED.md)
- [notebook details](notebooks/README_DETAILED.md)
- [test details](tests/README_DETAILED.md)
