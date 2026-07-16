"""Run one explicit BioEmu worker shard."""

from __future__ import annotations

import argparse
from pathlib import Path

from parallel_bioemu_tools.colab_gpu import ensure_allowed_gpu
from parallel_bioemu_tools.hf_storage import HuggingFaceStorage
from parallel_bioemu_tools.colab_runtime import release_colab_runtime
from parallel_bioemu_tools.manifests import WorkerConfig, read_json, shard_assignment_from_dict
from parallel_bioemu_tools.worker import run_worker_shard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-manifest", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument(
        "--hf-repo-id",
        required=True,
        help="Hugging Face dataset repository that will receive this shard.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--worker-id", default=None, help="Stable worker identity; defaults to the shard id.")
    parser.add_argument("--heartbeat-interval-seconds", type=float, default=120.0)
    parser.add_argument(
        "--release-colab-runtime",
        action="store_true",
        help="Release the managed Colab runtime after every task is remotely verified.",
    )
    parser.add_argument(
        "--allowed-gpu",
        dest="allowed_gpu_types",
        action="append",
        default=None,
        help="Allowed GPU type token. Repeat to allow multiple types; defaults to L4 and T4.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shard = shard_assignment_from_dict(read_json(args.shard_manifest))
    gpu_report = ensure_allowed_gpu(tuple(args.allowed_gpu_types or ("L4", "T4")))
    print(gpu_report.describe())
    storage = HuggingFaceStorage(repo_id=args.hf_repo_id)
    results = run_worker_shard(
        args.shard_manifest,
        storage=storage,
        config=WorkerConfig(
            workspace_root=args.workspace_root,
            overwrite=args.overwrite,
            chunk_size=args.chunk_size,
            heartbeat_interval_seconds=args.heartbeat_interval_seconds,
            worker_id=args.worker_id or shard.shard_id,
            enable_heartbeat=True,
        ),
    )
    print(f"{shard.shard_id}: {len(results)} tasks completed")
    if args.release_colab_runtime:
        print("All shard outputs are remotely verified; releasing the Colab runtime.", flush=True)
        if not release_colab_runtime():
            print("Not running in hosted Colab; runtime release was skipped.")


if __name__ == "__main__":
    main()
