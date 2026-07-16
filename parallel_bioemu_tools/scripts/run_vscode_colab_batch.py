"""Run a resumable multi-protein batch on one VS Code Colab assignment."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_SOURCE = ROOT / "parallel_bioemu_tools/src"
for import_path in (ROOT, PACKAGE_SOURCE):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from guided_conformational_path.colab_transfer import create_workspace_archive
from parallel_bioemu_tools.vscode_batch import (
    ColabBatchJob,
    run_vscode_colab_batch,
)
from parallel_bioemu_tools.vscode_colab import (
    ColabJupyterClient,
    read_vscode_safe_storage_password,
    wait_for_assigned_server,
)


def parse_args() -> argparse.Namespace:
    """Parse batch targets and runtime lifecycle options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        metavar="PROTEIN_ID/PAIR_ID",
        help="Protein and endpoint-pair identifier; repeat for multiple jobs.",
    )
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--required-gpu", default="L4")
    parser.add_argument("--release-on-success", action="store_true")
    parser.add_argument("--wait-for-assignment-seconds", type=float, default=15 * 60)
    parser.add_argument(
        "--recover-kernel-id",
        help="Wait for a previously disconnected controller kernel before resuming.",
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--workspace-archive",
        type=Path,
        default=ROOT / "outputs/colab_transfers/guided_conformation_workspace.zip",
    )
    return parser.parse_args()


def parse_job(value: str, n_samples: int) -> ColabBatchJob:
    """Parse ``PROTEIN_ID/PAIR_ID`` into one immutable batch job."""
    parts = value.split("/", maxsplit=1)
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError(
            f"Expected PROTEIN_ID/PAIR_ID, received {value!r}"
        )
    return ColabBatchJob(parts[0], parts[1], n_samples)


def main() -> None:
    """Package, connect, run, import, verify, and conditionally release."""
    args = parse_args()
    root = args.repo_root.expanduser().resolve()
    jobs = [parse_job(value, args.n_samples) for value in args.pair]
    print("Building current workspace archive...", flush=True)
    create_workspace_archive(
        root,
        root / "data/dataset/first_five_proteins",
        args.workspace_archive,
    )

    print(
        "Reading the VS Code Colab assignment from macOS Keychain "
        "(at most one authorization prompt)...",
        flush=True,
    )
    password = read_vscode_safe_storage_password()
    try:
        print(
            f"Waiting up to {args.wait_for_assignment_seconds:g} seconds for an "
            f"assigned {args.required_gpu} server...",
            flush=True,
        )
        server = wait_for_assigned_server(
            password,
            args.required_gpu,
            timeout_seconds=args.wait_for_assignment_seconds,
        )
    finally:
        del password
    print(
        f"Selected assigned server: {server.label or 'unnamed'} "
        f"({server.accelerator} {server.variant})",
        flush=True,
    )

    client = ColabJupyterClient(server)
    if args.recover_kernel_id:
        print(
            f"Checking prior controller kernel {args.recover_kernel_id} before recovery...",
            flush=True,
        )
        client.wait_for_kernel_idle(args.recover_kernel_id)

    results = run_vscode_colab_batch(
        client,
        jobs,
        repo_root=root,
        workspace_archive=args.workspace_archive,
        required_gpu=args.required_gpu,
        release_on_success=args.release_on_success,
    )
    print("\nBATCH_SUMMARY_JSON=" + json.dumps([asdict(item) for item in results], indent=2))


if __name__ == "__main__":
    main()
