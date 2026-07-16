"""Resumable serial BioEmu batches on one VS Code-assigned Colab runtime."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from guided_conformational_path.colab_transfer import import_result_archive


class BatchControllerError(RuntimeError):
    """Raised when a safe batch operation or local verification fails."""


class ColabBatchClient(Protocol):
    """Operations the batch controller requires from a Colab client."""

    def status(self) -> dict[str, Any]: ...

    def upload(self, source: str | Path, remote_name: str) -> None: ...

    def create_kernel(self) -> str: ...

    def remote_file_exists(self, remote_path: str) -> bool: ...

    def download(self, remote_path: str, destination: str | Path) -> Path: ...

    def execute(
        self,
        kernel_id: str,
        code: str,
        label: str,
        *,
        allow_disconnect: bool = False,
        timeout_seconds: float = 8 * 60 * 60,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class ColabBatchJob:
    """One protein endpoint pair to run and import."""

    protein_id: str
    pair_id: str
    n_samples: int = 200

    def __post_init__(self) -> None:
        for field_name, value in (
            ("protein_id", self.protein_id),
            ("pair_id", self.pair_id),
        ):
            if not value or value in {".", ".."} or "/" in value or "\\" in value:
                raise ValueError(f"{field_name} must be a safe path component")
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive")

    @property
    def run_name(self) -> str:
        """Return the sample-count-specific run directory name."""
        return f"n_{self.n_samples}"

    @property
    def archive_name(self) -> str:
        """Return the canonical result archive filename."""
        return f"{self.protein_id}__{self.pair_id}__{self.run_name}.zip"

    @property
    def remote_archive_path(self) -> str:
        """Return the result path on the Colab filesystem."""
        return f"/content/exports/{self.archive_name}"


@dataclass(frozen=True, slots=True)
class BatchJobResult:
    """Verified local outcome for one batch job."""

    protein_id: str
    pair_id: str
    requested_samples: int
    retained_samples: int
    graph_nodes: int
    graph_edges: int
    path_node_count: int
    total_cost: float
    source: str


def run_vscode_colab_batch(
    client: ColabBatchClient,
    jobs: list[ColabBatchJob],
    *,
    repo_root: str | Path,
    workspace_archive: str | Path,
    required_gpu: str = "L4",
    release_on_success: bool = False,
) -> list[BatchJobResult]:
    """Run missing jobs sequentially and release only after all verify locally.

    Recovery precedence is: verified canonical local output, an already
    downloaded local archive, a completed remote archive, and finally a new or
    resumed remote pair run. Every result is imported and verified before the
    next protein starts.
    """
    if not jobs:
        raise ValueError("jobs must contain at least one protein pair")
    root = Path(repo_root).expanduser().resolve()
    archive = Path(workspace_archive).expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"Workspace archive not found: {archive}")

    _wait_for_runtime(client)
    pending = [job for job in jobs if verify_local_job(root, job) is None]
    kernel_id: str | None = None
    if pending:
        print(f"Uploading one workspace archive for {len(pending)} pending job(s)...", flush=True)
        client.upload(archive, "/content/guided_conformation_workspace.zip")
        kernel_id = client.create_kernel()
        print(f"Created dedicated Colab kernel: {kernel_id}", flush=True)
        client.execute(
            kernel_id,
            _remote_extract_code(required_gpu),
            f"Verify {required_gpu} and extract workspace",
        )
        client.execute(
            kernel_id,
            _remote_install_code(),
            "Install project and BioEmu CUDA dependencies",
            timeout_seconds=30 * 60,
        )

    results: list[BatchJobResult] = []
    try:
        for index, job in enumerate(jobs, start=1):
            print(
                f"\nBatch job {index}/{len(jobs)}: {job.protein_id} / {job.pair_id}",
                flush=True,
            )
            verified = verify_local_job(root, job)
            if verified is not None:
                print("Verified local output already exists; skipping.", flush=True)
                results.append(_result_from_summary(verified, job, "existing_local_output"))
                continue

            local_archive = root / "outputs/colab_transfers/downloads" / job.archive_name
            if local_archive.is_file():
                print(f"Importing existing local archive: {local_archive}", flush=True)
                _import_archive(local_archive, root, job)
                results.append(_verified_result(root, job, "existing_local_archive"))
                continue

            if kernel_id is None:
                raise BatchControllerError("A pending job has no active controller kernel")
            if client.remote_file_exists(job.remote_archive_path):
                print(
                    f"Recovering completed remote archive: {job.remote_archive_path}",
                    flush=True,
                )
            else:
                client.execute(
                    kernel_id,
                    _remote_pair_code(job),
                    f"BioEmu and graph: {job.protein_id} / {job.pair_id}",
                )

            # The Colab Contents API can lag behind files created by a kernel.
            # Validate through that same kernel before using the file endpoint.
            client.execute(
                kernel_id,
                _remote_validate_archive_code(job),
                f"Validate remote archive: {job.protein_id} / {job.pair_id}",
                timeout_seconds=30 * 60,
            )

            print(f"Downloading {job.archive_name}...", flush=True)
            client.download(job.remote_archive_path, local_archive)
            _import_archive(local_archive, root, job)
            results.append(_verified_result(root, job, "remote_colab"))
    except Exception:
        print(
            "Batch did not fully verify; leaving the Colab runtime connected for recovery.",
            flush=True,
        )
        raise

    if release_on_success:
        if kernel_id is None:
            kernel_id = client.create_kernel()
        print("All requested jobs verified locally; releasing the Colab runtime.", flush=True)
        client.execute(
            kernel_id,
            "from google.colab import runtime\n"
            "print('RELEASE_REQUESTED: verified batch is durable locally.', flush=True)\n"
            "runtime.unassign()\n",
            "Release Colab runtime",
            allow_disconnect=True,
            timeout_seconds=120,
        )
    return results


def verify_local_job(
    repo_root: str | Path,
    job: ColabBatchJob,
) -> dict[str, Any] | None:
    """Return a verified graph summary, or ``None`` when no output exists."""
    root = Path(repo_root).expanduser().resolve()
    bundle_dir = (
        root
        / "data/bioemu_outputs/first_five_proteins"
        / job.protein_id
        / job.pair_id
        / job.run_name
    )
    graph_dir = (
        root / "outputs/dataset_bioemu" / job.protein_id / job.pair_id / job.run_name
    )
    if not bundle_dir.exists() and not graph_dir.exists():
        return None
    required = (
        bundle_dir / "manifest.json",
        graph_dir / "experiment_summary.json",
        graph_dir / "path_summary.json",
        graph_dir / "graph.png",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise BatchControllerError(
            "Partial local output exists; refusing to overwrite it automatically. Missing: "
            + ", ".join(missing)
        )
    bundle_manifest = _read_json(required[0])
    graph_summary = _read_json(required[1])
    path_summary = _read_json(required[2])
    graph_input = graph_summary.get("input", {})
    if graph_input.get("protein_id") != job.protein_id:
        raise BatchControllerError("Local graph protein_id does not match the requested job")
    if graph_input.get("pair_id") != job.pair_id:
        raise BatchControllerError("Local graph pair_id does not match the requested job")
    if int(bundle_manifest.get("n_samples_requested", 0)) != job.n_samples:
        raise BatchControllerError("Local BioEmu requested-sample count does not match the job")
    samples = bundle_manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise BatchControllerError("Local BioEmu bundle contains no retained samples")
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or not sample.get("pdb_path"):
            raise BatchControllerError(f"Local BioEmu sample {index} has no PDB path")
        sample_path = bundle_dir / str(sample["pdb_path"])
        if not sample_path.is_file() or sample_path.stat().st_size == 0:
            raise BatchControllerError(f"Local BioEmu sample PDB is missing: {sample_path}")
    node_ids = path_summary.get("node_ids")
    if not isinstance(node_ids, list) or len(node_ids) < 2:
        raise BatchControllerError("Local shortest path is missing endpoint nodes")
    if node_ids[0] != "A" or node_ids[-1] != "B":
        raise BatchControllerError("Local shortest path does not run from A to B")
    if required[3].stat().st_size == 0:
        raise BatchControllerError("Local graph PNG is empty")
    if int(graph_summary.get("n_generated_nodes", -1)) != len(samples):
        raise BatchControllerError("Graph node count disagrees with retained BioEmu samples")
    total_cost = float(path_summary.get("total_cost", float("nan")))
    if not math.isfinite(total_cost) or total_cost < 0:
        raise BatchControllerError("Local shortest-path cost is not finite and non-negative")
    return graph_summary


def _wait_for_runtime(
    client: ColabBatchClient,
    *,
    timeout_seconds: float = 10 * 60,
    poll_seconds: float = 10,
) -> dict[str, Any]:
    """Wait through Colab allocation warm-up until Jupyter answers."""
    deadline = time.monotonic() + timeout_seconds
    attempt = 0
    while True:
        attempt += 1
        try:
            return client.status()
        except Exception:
            if time.monotonic() >= deadline:
                raise
            print(
                f"Colab Jupyter endpoint is not ready (attempt {attempt}); retrying...",
                flush=True,
            )
            time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))


def _import_archive(archive: Path, root: Path, job: ColabBatchJob) -> None:
    """Import one archive and cross-check its identity."""
    imported = import_result_archive(archive, root)
    if imported["protein_id"] != job.protein_id or imported["pair_id"] != job.pair_id:
        raise BatchControllerError("Imported archive identity does not match the requested job")
    if int(imported["requested_samples"]) != job.n_samples:
        raise BatchControllerError("Imported archive sample count does not match the requested job")


def _verified_result(root: Path, job: ColabBatchJob, source: str) -> BatchJobResult:
    summary = verify_local_job(root, job)
    if summary is None:
        raise BatchControllerError("Imported result did not produce canonical local outputs")
    return _result_from_summary(summary, job, source)


def _result_from_summary(
    summary: dict[str, Any],
    job: ColabBatchJob,
    source: str,
) -> BatchJobResult:
    """Normalize saved experiment summary fields for the controller report."""
    path = summary.get("path", {})
    graph = summary.get("graph", {})
    node_ids = path.get("node_ids") or summary.get("path_node_ids") or []
    total_cost = path.get("total_cost", summary.get("total_cost", 0.0))
    return BatchJobResult(
        protein_id=job.protein_id,
        pair_id=job.pair_id,
        requested_samples=job.n_samples,
        retained_samples=int(summary.get("n_generated_nodes", 0)),
        graph_nodes=int(graph.get("n_nodes", summary.get("n_nodes", 0))),
        graph_edges=int(graph.get("n_edges", summary.get("n_edges", 0))),
        path_node_count=len(node_ids),
        total_cost=float(total_cost),
        source=source,
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BatchControllerError(f"Expected a JSON object: {path}")
    return value


def _remote_extract_code(required_gpu: str) -> str:
    """Return remote source that validates the GPU and extracts the workspace."""
    return f"""
from pathlib import Path
import json
import shutil
import subprocess
import zipfile

CONTENT_ROOT = Path('/content')
UPLOAD_ARCHIVE = CONTENT_ROOT / 'guided_conformation_workspace.zip'
WORKSPACE_DIR = CONTENT_ROOT / 'protein-conformation-guided-random-walk'
gpu_name = subprocess.check_output(
    ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True
).strip()
print('Connected GPU:', gpu_name, flush=True)
if {required_gpu.upper()!r} not in gpu_name.upper():
    raise RuntimeError(
        'This batch requires {required_gpu.upper()}; connected GPU is ' + repr(gpu_name)
    )
if not UPLOAD_ARCHIVE.is_file():
    raise FileNotFoundError(UPLOAD_ARCHIVE)
if WORKSPACE_DIR.exists():
    shutil.rmtree(WORKSPACE_DIR)
with zipfile.ZipFile(UPLOAD_ARCHIVE) as handle:
    seen = set()
    for member in handle.infolist():
        parts = Path(member.filename).parts
        unix_mode = member.external_attr >> 16
        if (not parts or Path(member.filename).is_absolute() or '..' in parts
                or member.filename in seen or (unix_mode & 0o170000) == 0o120000):
            raise RuntimeError('Unsafe workspace archive member: ' + repr(member.filename))
        seen.add(member.filename)
    workspace_manifest = json.loads(handle.read('colab_workspace_manifest.json'))
    if workspace_manifest.get('schema_version') != 'guided_conformation_colab_workspace_v1':
        raise RuntimeError('Unsupported workspace archive schema')
    handle.extractall(CONTENT_ROOT)
print('Extracted workspace:', WORKSPACE_DIR, flush=True)
""".strip()


def _remote_install_code() -> str:
    """Return remote source that installs BioEmu and the uploaded project."""
    return """
import subprocess
import sys

subprocess.run(
    [sys.executable, '-m', 'pip', 'uninstall', '-y',
     'tensorflow', 'tensorflow-cpu', 'tf-keras'],
    check=False,
)
subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-q', '-e',
     f'{WORKSPACE_DIR}[notebook]', 'bioemu[cuda]'],
    check=True,
)
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))
from guided_conformational_path.colab_pair import run_colab_pair
print('Project and BioEmu installation complete.', flush=True)
""".strip()


def _remote_pair_code(job: ColabBatchJob) -> str:
    """Return remote source for one pair, using JSON literals for safe quoting."""
    return (
        "import json\n"
        "from guided_conformational_path.colab_pair import run_colab_pair\n"
        f"result = run_colab_pair({json.dumps(job.pair_id)}, {job.n_samples})\n"
        "print('BATCH_RESULT_JSON=' + json.dumps(result, sort_keys=True), flush=True)\n"
    )


def _remote_validate_archive_code(job: ColabBatchJob) -> str:
    """Return remote source that verifies a completed archive before transfer."""
    return f"""
import hashlib
import json
import zipfile
from pathlib import Path

archive_path = Path({job.remote_archive_path!r})
if not archive_path.is_file() or archive_path.stat().st_size == 0:
    raise RuntimeError('Remote result archive is absent or empty: ' + str(archive_path))
with zipfile.ZipFile(archive_path) as handle:
    bad_member = handle.testzip()
    if bad_member is not None:
        raise RuntimeError('Corrupt ZIP member: ' + bad_member)
    manifest = json.loads(handle.read('result_manifest.json'))
if manifest.get('schema_version') != 'guided_conformation_colab_result_v1':
    raise RuntimeError('Unexpected remote result schema')
if manifest.get('protein_id') != {job.protein_id!r}:
    raise RuntimeError('Remote result protein_id mismatch')
if manifest.get('pair_id') != {job.pair_id!r}:
    raise RuntimeError('Remote result pair_id mismatch')
if int(manifest.get('requested_samples', 0)) != {job.n_samples}:
    raise RuntimeError('Remote result sample-count mismatch')
digest = hashlib.sha256()
with archive_path.open('rb') as handle:
    for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
        digest.update(chunk)
print('REMOTE_ARCHIVE_VERIFIED', archive_path.stat().st_size, digest.hexdigest(), flush=True)
""".strip()
