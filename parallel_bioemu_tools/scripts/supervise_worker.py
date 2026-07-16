"""Retry one immutable worker shard on the currently attached runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from parallel_bioemu_tools.colab_runtime import release_colab_runtime
from parallel_bioemu_tools.hf_storage import HuggingFaceStorage
from parallel_bioemu_tools.supervisor import SupervisorConfig, run_supervised_shard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-manifest", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument(
        "--hf-repo-id",
        required=True,
        help="Hugging Face dataset repository that will receive this shard.",
    )
    parser.add_argument(
        "--allowed-gpu",
        dest="allowed_gpu_types",
        action="append",
        default=None,
        help="Allowed GPU type token. Repeat to allow multiple types; defaults to L4 and T4.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--worker-id", default=None, help="Stable worker identity; defaults to the shard id.")
    parser.add_argument("--heartbeat-interval-seconds", type=float, default=120.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=30.0)
    parser.add_argument(
        "--release-colab-runtime",
        action="store_true",
        help="Release the managed Colab runtime after all shard tasks are remotely verified.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    storage = HuggingFaceStorage(repo_id=args.hf_repo_id)
    shard_id = json.loads(args.shard_manifest.read_text(encoding="utf-8")).get("shard_id")
    result = run_supervised_shard(
        args.shard_manifest,
        storage=storage,
        config=SupervisorConfig(
            workspace_root=args.workspace_root,
            overwrite=args.overwrite,
            max_attempts=args.max_attempts,
            retry_delay_seconds=args.retry_delay_seconds,
            allowed_gpu_types=tuple(args.allowed_gpu_types or ("L4", "T4")),
            chunk_size=args.chunk_size,
            worker_id=args.worker_id or shard_id,
            heartbeat_interval_seconds=args.heartbeat_interval_seconds,
            enable_heartbeat=True,
        ),
    )
    print(result.gpu_report.describe())
    print(
        f"Shard completed after {result.attempts_used} attempt(s); "
        f"remotely verified {len(result.completions)} task(s)."
    )
    if args.release_colab_runtime:
        print("All shard outputs are remotely verified; releasing the Colab runtime.", flush=True)
        if not release_colab_runtime():
            print("Not running in hosted Colab; runtime release was skipped.")


if __name__ == "__main__":
    main()
