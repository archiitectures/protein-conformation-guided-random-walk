# Codex Handoff

## Project Purpose

Protein Conformation Guided-Random Walk is a research prototype for proposing
candidate conformational transition paths between two protein endpoint
structures. It treats generated structures as graph nodes and heuristic
structural scores as graph-edge costs. A shortest or ranked path is a proposed
structural path, not a validated physical transition pathway.

The project intentionally does **not** claim that equilibrium-like generated
samples establish kinetic connectivity, transition probabilities, MFPTs, or a
true molecular mechanism.

## Current State

- Main branch before this handoff was `44002e9`.
- The working tree includes the dataset namespace, Colab transfer, controller,
  documentation, and notebook changes for the non-ideal-transition experiment.
- Environment management uses `uv`; the project can be run locally on macOS,
  while BioEmu sampling is performed on Linux/Colab GPU runtimes.
- Generated datasets and experiment outputs are intentionally ignored by Git.

## Core Components

- `guided_conformational_path/`: graph nodes, generators, scoring, validity,
  graph building, path search, visualization, PDB IO, BioEmu adapters, and
  Colab transfer helpers.
- `protein_conformer_dataset/`: UniProt/PDB endpoint dataset construction,
  residue alignment, quality control, manifests, and exports.
- `parallel_bioemu_tools/`: VS Code Colab extension integration, serial batch
  controller, recovery behavior, and result collection helpers.
- `scripts/`: command-line data build, transfer, and experiment utilities.
- `notebooks/`: notebook UIs for synthetic experiments, PDB endpoint runs,
  BioEmu Colab sampling, and local result analysis.
- `data/input/`: input target manifests.

## Completed Experiments

### First five proteins

The earlier proof-of-concept data and outputs use the legacy namespace
`first_five_proteins`. Legacy graph outputs remain at:

`outputs/dataset_bioemu/<protein_id>/<pair_id>/n_<samples>/`

### Non-ideal transitions (`n=3`)

Target manifest:

`data/input/non_ideal_transitions_n3.csv`

Targets:

| Protein | UniProt | Endpoints | Special handling |
| --- | --- | --- | --- |
| Human Mad2 | Q13257 | 1DUJ A -> 1S2H A | Full aligned endpoint pair |
| Thermosynechococcus KaiB | Q79V61 | 2QKE A -> 5JYT A | Exact aligned residue subset for mutation-tolerant graph comparison; BioEmu uses canonical UniProt sequence |
| E. coli RfaH CTD | P0AFW0 | 2OUG A -> 2LCL A | CTD-only alignment-derived domain selection; removes the 2LCL construct tag |

The batch completed on a T4 runtime with 200 requested samples per target.
Results are local, but ignored by Git:

- BioEmu bundles: `data/bioemu_outputs/non_ideal_transitions_n3/`
- Graph results: `outputs/dataset_bioemu/non_ideal_transitions_n3/`
- Colab run status: `outputs/colab_transfers/non_ideal_transitions_n3_status.json`

Recorded results:

| Protein | Retained samples | Graph nodes | Graph edges | Best-path nodes | Cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mad2 | 176 | 178 | 1,623 | 5 | 109.52 |
| KaiB | 146 | 148 | 1,188 | 4 | 90.61 |
| RfaH CTD | 168 | 170 | 1,390 | 5 | 112.75 |

The successful three-protein batch took approximately 40 minutes 35 seconds,
including setup, sampling, graph construction, import, and verification.

## Colab Workflow

1. Build an endpoint dataset locally with `scripts/build_dataset.py`.
2. Package a named dataset directory with `scripts/colab_transfer.py` or the
   VS Code batch script.
3. Attach a GPU in the VS Code Colab extension. BioEmu requires Linux; T4 is
   supported for lower-cost runs and L4 is supported for faster runs.
4. Run `parallel_bioemu_tools/scripts/run_vscode_colab_batch.py` with:
   - `--dataset-label <label>`
   - one or more `--pair PROTEIN_ID/PAIR_ID`
   - `--required-gpu T4` or `L4`
   - `--release-on-success`
5. The controller packages the current workspace, runs remote sampling and
   graph construction sequentially, downloads the result archive, imports it
   locally, verifies it, and only then releases the Colab runtime.

For a disconnected controller kernel, use `--recover-kernel-id <id>`. This
reuses the prepared remote kernel and avoids another archive upload/install.

The controller writes non-secret progress JSON via `--status-file`; its phases
are `building_workspace`, `awaiting_t4_assignment`, `t4_assigned`,
`running_batch`, `complete`, or `failed`.

## Important Caveats

- C-alpha geometry and heuristic edge scores are useful for ranking candidate
  structural paths, not physical transition validation.
- BioEmu produces plausible ensemble nodes; it does not by itself provide
  calibrated kinetics or prove graph-edge connectivity.
- Protein constructs, missing residues, engineered mutations, and sequence
  tags need explicit manifest/QC review before interpreting a path.
- Retained sample count can be below the requested count when sample parsing or
  endpoint correspondence filtering rejects structures.

## Deferred Reliability Work

These issues were identified in a read-only code review and should be addressed
before larger unattended datasets are run:

1. Persist and validate BioEmu seed and effective sampler kwargs before
   reusing an existing bundle. Current reuse is keyed principally by pair and
   sample count, so changed sampling configuration could reuse stale samples.
2. Add a dataset-pair manifest fingerprint to BioEmu bundles and graph outputs.
   `verify_local_job` should reject stale results if a dataset is rebuilt in
   place with changed residue mappings or selected sequence.
3. Add archive extraction limits for member count, per-file size, total
   uncompressed size, and compression ratio in both local and remote handlers.
4. Make `parallel_bioemu_tools` collection resumable: skip complete verified
   outputs and fail only on partial/invalid output state.
5. Distinguish fresh downloads/parses from checkpoint reuse in dataset build
   run summaries.

Do not modify active result directories in place while a Colab batch is
running. Prefer a new dataset label for materially different endpoint selection
or sampling configuration until the fingerprint work is complete.

## Validation

Typical local validation commands:

```bash
uv run pytest
UV_CACHE_DIR=.uv-cache uv --project parallel_bioemu_tools run pytest
python scripts/run_demo.py
```

## Resume Prompt

Paste the following into a new Codex chat from the repository root:

```text
Resume work on the Protein Conformation Guided-Random-Walk repository.

Read codex.md first, then inspect git status and the current README files.
Treat codex.md as the project handoff and preserve the scientific framing:
generated conformations are candidate nodes and graph paths are heuristic
structural proposals, not validated kinetic pathways.

The last completed experiment is `non_ideal_transitions_n3`; its generated
outputs are under `data/bioemu_outputs/non_ideal_transitions_n3` and
`outputs/dataset_bioemu/non_ideal_transitions_n3` if they are present locally.

Start by reviewing and implementing the deferred reliability work in codex.md:
sampling-config fingerprints, dataset-manifest fingerprints, safe ZIP limits,
resumable collector behavior, and accurate checkpoint telemetry. Do not rerun
BioEmu unless explicitly requested. Use uv for Python commands, retain the
separate dataset-label namespaces, run relevant tests, and do not revert user
changes in a dirty worktree.
```
