"""Download validated bundles and run local graph collection."""

from __future__ import annotations

import argparse
from pathlib import Path

from parallel_bioemu_tools.collector import collect_run
from parallel_bioemu_tools.hf_storage import HuggingFaceStorage
from parallel_bioemu_tools.manifests import CollectorConfig, read_json, run_manifest_from_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--hf-repo-id", default=None)
    parser.add_argument("--max-processes", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_manifest_from_dict(read_json(args.run_manifest))
    storage = HuggingFaceStorage(repo_id=args.hf_repo_id or manifest.hf_repo_id)
    results = collect_run(
        args.run_manifest,
        storage=storage,
        config=CollectorConfig(
            download_root=args.download_root,
            graph_root=args.graph_root,
            max_processes=args.max_processes,
            overwrite=args.overwrite,
        ),
    )
    print(f"collected {len(results)} result bundles")


if __name__ == "__main__":
    main()
