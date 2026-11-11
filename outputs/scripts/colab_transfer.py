"""Package local Colab inputs or import one downloaded pair result archive."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guided_conformational_path.colab_transfer import (
    create_workspace_archive,
    import_result_archive,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    """Parse Colab workspace packaging and result import commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    package = subparsers.add_parser("package-workspace", help="Build the no-Drive upload ZIP.")
    package.add_argument("--repo-root", type=Path, default=ROOT)
    package.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "data/dataset/first_five_proteins",
        help="Exported dataset directory, for example data/dataset/non_ideal_transitions_n3.",
    )
    package.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/colab_transfers/guided_conformation_workspace.zip",
    )

    import_result = subparsers.add_parser(
        "import-result",
        help="Import one result ZIP downloaded from the Colab extension.",
    )
    import_result.add_argument("archive", type=Path)
    import_result.add_argument("--repo-root", type=Path, default=ROOT)
    import_result.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Execute the selected transfer command and print machine-readable output."""
    args = parse_args()
    if args.command == "package-workspace":
        archive = create_workspace_archive(args.repo_root, args.dataset_root, args.output)
        result = {
            "archive": str(archive),
            "size_bytes": archive.stat().st_size,
            "sha256": sha256_file(archive),
        }
    else:
        result = import_result_archive(args.archive, args.repo_root, force=args.force)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
