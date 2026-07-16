"""Create immutable run and shard manifests for parallel BioEmu workers."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from parallel_bioemu_tools.manifests import PairRunSpec, RunManifest, write_json
from parallel_bioemu_tools.sharding import assign_cost_balanced_shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--hf-repo-id",
        required=True,
        help="Hugging Face dataset repository shared by workers and collector.",
    )
    parser.add_argument("--pairs-json", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.pairs_json.read_text(encoding="utf-8"))
    pairs = [PairRunSpec(**item) for item in payload]
    shards = assign_cost_balanced_shards(pairs, run_id=args.run_id, num_shards=args.num_shards)
    run_manifest = RunManifest(
        run_id=args.run_id,
        hf_repo_id=args.hf_repo_id,
        pairs=tuple(pairs),
        shards=tuple(shards),
    )
    write_json(args.run_manifest, asdict(run_manifest))
    args.shards_dir.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        write_json(args.shards_dir / f"{shard.shard_id}.json", asdict(shard))
    print(args.run_manifest)


if __name__ == "__main__":
    main()
