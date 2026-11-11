# parallel_bioemu_tools

Same-worktree orchestration utilities for parallel BioEmu bundle generation and
local collection. This directory is a subproject of the root repository under
`aa_overall_proj_1`; its separate `uv.lock` isolates dependencies, not source
control or filesystem layout.

## Scope

- Explicit immutable run and shard manifests.
- Provider-neutral worker flow for Colab now and other remote workers later.
- Optional Hugging Face Hub storage through a small injectable backend.
- Local collector that downloads validated bundles and runs existing graph APIs with bounded process concurrency.

This subproject does not start live BioEmu or Hugging Face operations by itself during tests.

## Layout

- `src/parallel_bioemu_tools`: package code.
- `scripts/prepare_run.py`: build run + shard manifests.
- `scripts/run_worker.py`: execute one shard.
- `scripts/collect_results.py`: download validated bundles and run local graph analysis.
- `scripts/watch_run.py`: macOS-only heartbeat inspection and recovery queue writer.
- Colab worker execution is provided by the CLI scripts; there is no checked-in
  worker notebook in this subproject.
- `tests/`: isolated tests with mocked external operations.

## Commands

Create the subproject environment from its checked-in lockfile:

```bash
UV_CACHE_DIR=.uv-cache uv --project parallel_bioemu_tools sync \
  --extra dev --extra huggingface
```

Prepare a run:

```bash
UV_CACHE_DIR=.uv-cache uv --project parallel_bioemu_tools run \
  python parallel_bioemu_tools/scripts/prepare_run.py \
  --run-id demo_run \
  --hf-repo-id your-account/your-private-dataset \
  --pairs-json /path/to/pairs.json \
  --run-manifest parallel_bioemu_tools/tmp/demo/run_manifest.json \
  --shards-dir parallel_bioemu_tools/tmp/demo/shards
```

Run one shard:

```bash
HF_TOKEN=... UV_CACHE_DIR=.uv-cache uv --project parallel_bioemu_tools run \
  python parallel_bioemu_tools/scripts/run_worker.py \
  --shard-manifest parallel_bioemu_tools/tmp/demo/shards/shard_000.json \
  --workspace-root parallel_bioemu_tools/tmp/demo/worker \
  --hf-repo-id your-account/your-private-dataset \
  --chunk-size 20 --worker-id worker-000 --heartbeat-interval-seconds 120
```

Workers accept either `L4` or `T4` by default. Override the policy with
repeated `--allowed-gpu` flags when you need a narrower list.

For bounded unattended retries on the current runtime, use:

```bash
HF_TOKEN=... UV_CACHE_DIR=.uv-cache uv --project parallel_bioemu_tools run \
  python parallel_bioemu_tools/scripts/supervise_worker.py \
  --shard-manifest parallel_bioemu_tools/tmp/demo/shards/shard_000.json \
  --workspace-root parallel_bioemu_tools/tmp/demo/worker \
  --hf-repo-id your-account/your-private-dataset \
  --max-attempts 3 \
  --retry-delay-seconds 30 --chunk-size 20 \
  --worker-id worker-000 --heartbeat-interval-seconds 120
```

In hosted Colab, add `--release-colab-runtime`. Runtime release occurs only after all files and
the final completion marker are visible in the Hugging Face repository. Any
generation, upload, or verification failure leaves the runtime attached for
inspection and retry.

For a small serial batch on one server already assigned by the VS Code Colab
extension, select the requested GPU in VS Code and run:

```bash
UV_CACHE_DIR=.uv-cache uv --project parallel_bioemu_tools run \
  python parallel_bioemu_tools/scripts/run_vscode_colab_batch.py \
  --pair P00889/1CTS_A__2CTS_A \
  --pair P35557/1V4T_A__1V4S_A \
  --n-samples 200 \
  --required-gpu L4 \
  --release-on-success
```

This controller reads the VS Code safe-storage key from macOS Keychain once
and keeps it only in process memory. It uploads and installs once, verifies and
imports each protein before continuing, recovers completed remote archives,
and releases the runtime only after every requested result is durable locally.

Collect results locally:

```bash
HF_TOKEN=... UV_CACHE_DIR=.uv-cache uv --project parallel_bioemu_tools run \
  python parallel_bioemu_tools/scripts/collect_results.py \
  --run-manifest parallel_bioemu_tools/tmp/demo/run_manifest.json \
  --download-root parallel_bioemu_tools/tmp/demo/downloads \
  --graph-root parallel_bioemu_tools/tmp/demo/graphs \
  --max-processes 2
```

## Exact overnight workflow

1. On macOS, prepare one immutable run manifest and shard manifests with
   `prepare_run.py`. Choose a chunk size such as 20 samples when a worker can
   disappear during the night; a 45-sample task becomes 20, 20, and 5. Durable
   raw chunks are stored under
   `run_id/_chunks/protein_id/pair_id/n_<total>/chunk_<index>_n_<size>/`, separate
   from the final collector-facing bundle. A completed chunk is durable only
   after its bundle files and `_chunk_completed.json` marker are verified.
2. Authenticate to Google/Colab and start the Linux/CUDA workers manually.
   Launch each worker or `supervise_worker.py` with a stable worker identity,
   chunk size and stable worker identity. Workers publish heartbeats under
   `run_id/_heartbeats/<shard_id>.json`. Google authentication and runtime
   allocation remain manual operations.
3. On macOS, run the compute-free watchdog in a separate terminal:

```bash
UV_CACHE_DIR=.uv-cache uv --project parallel_bioemu_tools run \
  python parallel_bioemu_tools/scripts/watch_run.py \
  --run-manifest parallel_bioemu_tools/tmp/demo/run_manifest.json \
  --status-path parallel_bioemu_tools/tmp/demo/watch_status.json \
  --recovery-queue parallel_bioemu_tools/tmp/demo/recovery_queue.json \
  --hf-repo-id your-account/your-private-dataset \
  --poll-interval 300 --stale-after 900
```

The watchdog reads the existing `RunManifest` shard list and requires
`HF_TOKEN` through `HuggingFaceStorage`; it checks the remote heartbeat prefix,
not an invented local worker list. Use `--once` for automation or a scheduled
check. Missing, malformed, or stale heartbeats (900 seconds by default) add
local recovery records; the watchdog does not allocate Colab, authenticate
Google, or start BioEmu. Resume workers from the queue and let completion
markers skip already durable chunks. BioEmu cannot resume inside a 20-sample
chunk, so a worker death during a chunk regenerates that entire chunk only.
Shards whose canonical `_worker_completed.json` markers are already visible
are excluded from recovery, and a terminal `completed` heartbeat is not later
misclassified as stale.
4. After all chunk completion markers are remotely visible, run
   `collect_results.py` on macOS. A resumed Linux worker downloads completed
   chunks, validates and merges them, and uploads one canonical bundle before
   writing `_worker_completed.json`. The Mac collector downloads that merged
   bundle and performs graph construction; BioEmu generation remains
   Linux/CUDA-only. Do not treat a fresh heartbeat as proof that a remote
   runtime is healthy.

The recovery queue is intentionally advisory. Colab runtime allocation and
Google authentication cannot be restarted reliably by this repository, so a
human must attach a replacement runtime and relaunch the same shard. Completed
20-sample chunks are then skipped. There is no Apple Silicon BioEmu fallback;
the local Mac coordinates recovery and graph analysis without pretending that
BioEmu supports MPS.

Run tests:

```bash
UV_CACHE_DIR=.uv-cache uv --project parallel_bioemu_tools run \
  --extra dev pytest parallel_bioemu_tools/tests -q
```

## Security

- `HF_TOKEN` is read only from the environment or an explicit secure argument to the storage client.
- Tokens are never persisted to manifests, logs, or output files.

## Durable Completion

For each task, the worker:

1. Generates or resumes the portable BioEmu bundle through the root package.
2. Validates all sample PDB paths.
3. Uploads the bundle and verifies every uploaded path remotely.
4. Uploads `_worker_completed.json` last and verifies it remotely.
5. Releases Colab only after every task in the shard reaches step 4.

The collector refuses bundles without a matching completion marker. This keeps
partial uploads out of graph construction and prevents successful-looking runs
from shutting down before their outputs are durable.

If a T4 runtime shuts down after step 4, the next runtime can reuse the remote
completion marker and skip that task without regenerating the bundle. If the
runtime dies before the marker upload, the shard is safe to retry because the
remote prefix is immutable and incomplete uploads are not treated as complete.

BioEmu still lacks mid-sample checkpointing. Chunking limits regeneration to
the active chunk, but a worker death during a chunk regenerates that whole
chunk. A task-level completion marker remains distinct from chunk markers.

The supervisor retries only while the current VM is alive. It cannot restart a
dead Colab VM, request a new GPU allocation, or automate Google account login;
those boundaries remain external to the notebook and CLI.

## Output paths

Run-scoped paths include:

- `run_id`
- `protein_id`
- `pair_id`
- `n_samples`

The default behavior is to refuse overwriting existing outputs.
`--hf-repo-id` is mandatory for workers so a shard cannot silently upload to a
repository different from the collector's run manifest.
