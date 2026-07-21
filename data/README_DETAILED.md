# Data Folder Walkthrough

`data/` contains small inputs and generated local state. Large or reproducible
outputs are intentionally not part of the source package.

## Inputs

- `input/endpoints.csv`: explicit endpoint-pair manifest used by
  `scripts/build_dataset.py`.
- `input/first_five_proteins.csv`: the five-pair manifest used by the VS Code
  Colab workflow.
- `input/non_ideal_transitions_n3.csv`: three deliberately harder endpoint
  comparisons. Its RfaH row is a CTD-only construct selected from full-length
  endpoint A by alignment to isolated endpoint B. Its KaiB row uses only exact
  matched endpoint residues while BioEmu samples the canonical sequence because
  the fold-switched endpoint is mutation-stabilized.
- `input/`: the stable place for additional endpoint manifests or one-UniProt-ID
  per-line files.
- `pdb_inputs/`: optional minimal PDB files for the legacy demo and exploratory
  PDB-endpoint notebook workflow. The checked-in fixtures are `a.pdb` and
  `b.pdb`; the demo accepts any paths supplied with `--endpoint-a` and
  `--endpoint-b`.
- `bioemu_inputs/`: optional FASTA provenance files for manually managed
  BioEmu runs.

The dataset generator's normal remote structure input is RCSB mmCIF, discovered
from exact UniProt mappings or named in the explicit manifest. It parses those
files with Gemmi, aligns endpoint sequences, and records residue mappings. The
minimal PDB reader under `guided_conformational_path/io.py` is a separate demo
utility and does not replace the dataset parser.

## Generated Dataset And Cache

- `cache/`: cached UniProt/RCSB responses and downloaded mmCIF structures.
- `dataset/`: standardized endpoint-pair exports. A pair directory contains
  raw and processed structures, aligned arrays, residue mapping, sequence
  choices, metadata, QC report, and `manifest.json`; the root contains CSV/YAML
  summaries and run logs.

The generated five-pair export is local ignored state, not a checked-in
dataset. The tracked input manifest is `input/first_five_proteins.csv`. Its
five current pair exports are `manual_review`; that status records QC warnings
for human interpretation and is not evidence of a transition.

Generated contents are ignored except for directory placeholders. Resume state
is under `<dataset-output>/.state/` and is reusable only when the input and
configuration fingerprint matches.

## BioEmu Results

The importer writes downloaded results under the dataset label recorded in the
result archive. The legacy five-pair location is:

```text
data/bioemu_outputs/first_five_proteins/<protein>/<pair>/n_<requested>/
  manifest.json
  nodes/*.pdb
  raw_bioemu/
```

For `non_ideal_transitions_n3`, the equivalent root is
`data/bioemu_outputs/non_ideal_transitions_n3/<protein>/<pair>/n_<requested>/`.

`manifest.json` is a portable `bioemu_bundle_v1` manifest. It records the
selected sequence, requested count, retained count, and relative node paths.
The requested count is not guaranteed to equal the retained count: trajectory
conversion uses BioEmu's retained physical trajectory frames and excludes
outputs labeled unphysical. The graph's C-alpha validity checks apply penalties;
they do not perform an additional node-filtering pass here. The local result
set may be incomplete: missing production archives, including P00889 and
P35557 when they have not been downloaded, are not represented by placeholder
bundles. Use
`scripts/colab_transfer.py import-result` to validate and install result ZIPs;
do not extract them manually.

## Outputs Boundary

Graph and visualization products belong under `outputs/`, not `data/`:

```text
outputs/dataset_bioemu/<protein>/<pair>/n_<requested>/
outputs/large_synthetic/
outputs/colab_transfers/
```

The parallel workflow uses separate run-scoped download and graph roots, plus
Hugging Face remote prefixes. See
`parallel_bioemu_tools/README.md` for those paths and completion markers.
