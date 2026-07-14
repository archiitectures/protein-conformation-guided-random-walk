# Data Folder Walkthrough

This folder is for user-provided inputs and small checked-in placeholders.

## `examples/`

Reserved for small example endpoint files.

Current placeholder:

```text
data/examples/.gitkeep
```

The quick demo can use files here:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/run_demo.py \
  --endpoint-a data/pdb_inputs/A.pdb \
  --endpoint-b data/pdb_inputs/B.pdb
```

## `pdb_inputs/`

Recommended folder for notebook PDB endpoint experiments.

Put your files here:

```text
data/pdb_inputs/A.pdb
data/pdb_inputs/B.pdb
```

The notebook PDB workflow expects those defaults.

## `bioemu_inputs/`

Recommended local record of FASTA files used for BioEmu sampling runs.

Typical file:

```text
data/bioemu_inputs/sequence.fasta
```

The local analyzer does not require this FASTA, but the Colab workflow and
notebooks reference it for provenance and reruns.

## `bioemu_outputs/`

Recommended location for downloaded or synced BioEmu bundles produced in Colab.

Typical layout:

```text
data/bioemu_outputs/latest/
  manifest.json
  nodes/
    bioemu_000000.pdb
    ...
  raw_bioemu/
```

The local BioEmu analysis CLI defaults to `data/bioemu_outputs/latest`.

## PDB Requirements

The current parser reads C-alpha atoms from fixed-column PDB records.

Supported:

- `ATOM` and `HETATM` records
- atom name `CA`
- chain ID
- residue number
- insertion code
- residue name
- XYZ coordinates

Not currently supported:

- full mmCIF parsing
- full atomistic structure handling
- alternate location logic
- sequence alignment
- robust structural alignment

## Matching Behavior

For PDB endpoint experiments, the loader first tries exact residue matching by:

```text
chain ID + residue number + insertion code
```

If exact overlap is too low and `matching_method="exact_then_geometric"`, it falls back to a heuristic C-alpha geometry match.

Use geometric matching only for exploratory tests with very similar backbones.

## Outputs Are Elsewhere

Generated outputs go under:

```text
outputs/
```

Do not put generated graphs, summaries, or node files in `data/`.
