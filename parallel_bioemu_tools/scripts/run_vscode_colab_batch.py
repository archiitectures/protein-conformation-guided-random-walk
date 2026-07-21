"""Run a resumable multi-protein batch on one VS Code Colab assignment."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
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
    parser.add_argument(
        "--dataset-label",
        default="first_five_proteins",
        help="Dataset directory name under data/dataset (default: first_five_proteins).",
    )
    parser.add_argument("--release-on-success", action="store_true")
    parser.add_argument("--wait-for-assignment-seconds", type=float, default=15 * 60)
    parser.add_argument(
        "--status-file",
        type=Path,
        default=ROOT / "outputs/colab_transfers/vscode_colab_batch_status.json",
        help="JSON status file updated throughout the batch lifecycle.",
    )
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
    try:
        _write_status(args.status_file, "building_workspace", jobs=jobs)
        print("Building current workspace archive...", flush=True)
        create_workspace_archive(
            root,
            root / "data/dataset" / args.dataset_label,
            args.workspace_archive,
        )

        _write_status(args.status_file, "awaiting_t4_assignment", jobs=jobs)
        print(
            "Reading the VS Code Colab assignment from macOS Keychain "
            "(at most one authorization prompt)...",
            flush=True,
        )
        password = read_vscode_safe_storage_password()
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
    except Exception as exc:
        _write_status(args.status_file, "failed", jobs=jobs, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if "password" in locals():
            del password
    print(
        f"Selected assigned server: {server.label or 'unnamed'} "
        f"({server.accelerator} {server.variant})",
        flush=True,
    )
    _write_status(
        args.status_file,
        "t4_assigned",
        jobs=jobs,
        server={"label": server.label, "accelerator": server.accelerator, "variant": server.variant},
    )

    client = ColabJupyterClient(server)
    if args.recover_kernel_id:
        print(
            f"Checking prior controller kernel {args.recover_kernel_id} before recovery...",
            flush=True,
        )
        client.wait_for_kernel_idle(args.recover_kernel_id)

    try:
        _write_status(args.status_file, "running_batch", jobs=jobs)
        results = run_vscode_colab_batch(
            client,
            jobs,
            repo_root=root,
            workspace_archive=args.workspace_archive,
            required_gpu=args.required_gpu,
            dataset_label=args.dataset_label,
            recover_kernel_id=args.recover_kernel_id,
            release_on_success=args.release_on_success,
        )
    except Exception as exc:
        _write_status(args.status_file, "failed", jobs=jobs, error=f"{type(exc).__name__}: {exc}")
        raise
    _write_status(args.status_file, "complete", jobs=jobs, results=[asdict(item) for item in results])
    print("\nBATCH_SUMMARY_JSON=" + json.dumps([asdict(item) for item in results], indent=2))


def _write_status(
    status_file: Path,
    phase: str,
    *,
    jobs: list[ColabBatchJob],
    **extra: object,
) -> None:
    """Write durable, non-secret controller progress for external monitoring."""
    output = status_file.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "jobs": [asdict(job) for job in jobs],
        **extra,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


if __name__ == "__main__":
    main()
