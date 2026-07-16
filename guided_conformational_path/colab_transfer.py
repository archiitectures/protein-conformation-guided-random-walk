"""Safe archives for no-Drive Colab upload and result download workflows."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


WORKSPACE_ARCHIVE_SCHEMA = "guided_conformation_colab_workspace_v1"
RESULT_ARCHIVE_SCHEMA = "guided_conformation_colab_result_v1"
WORKSPACE_ROOT = "protein-conformation-guided-random-walk"

_SOURCE_DIRECTORIES = (
    "guided_conformational_path",
    "protein_conformer_dataset",
    "scripts",
)
_SOURCE_FILES = (
    "pyproject.toml",
    "README.md",
)
_IGNORED_PARTS = {
    "__pycache__",
    ".DS_Store",
    ".git",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".uv-cache",
    ".venv",
}


class ColabTransferError(ValueError):
    """Raised when a Colab transfer archive is invalid or unsafe."""


def create_workspace_archive(
    repo_root: str | Path,
    dataset_root: str | Path,
    destination: str | Path,
) -> Path:
    """Create the minimal source-and-dataset ZIP uploaded to a Colab server.

    The archive intentionally excludes the local virtual environment, Git
    history, caches, tests, notebooks, and prior outputs. The notebook itself
    stays open locally in VS Code while its selected kernel executes remotely.
    """
    root = Path(repo_root).expanduser().resolve()
    dataset = Path(dataset_root).expanduser().resolve()
    output = Path(destination).expanduser().resolve()
    _require_directory(root, "repository root")
    _require_directory(dataset, "first-five dataset root")

    pair_manifests = sorted(dataset.glob("*/pairs/*/manifest.json"))
    if len(pair_manifests) != 5:
        raise ColabTransferError(
            f"Expected 5 pair manifests under {dataset}, found {len(pair_manifests)}"
        )

    sources: list[tuple[Path, PurePosixPath]] = []
    for directory_name in _SOURCE_DIRECTORIES:
        directory = root / directory_name
        _require_directory(directory, directory_name)
        sources.extend(_archive_files(directory, PurePosixPath(WORKSPACE_ROOT, directory_name)))
    for filename in _SOURCE_FILES:
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required workspace file not found: {path}")
        sources.append((path, PurePosixPath(WORKSPACE_ROOT, filename)))
    sources.extend(
        _archive_files(
            dataset,
            PurePosixPath(WORKSPACE_ROOT, "data", "dataset", "first_five_proteins"),
        )
    )

    manifest = {
        "schema_version": WORKSPACE_ARCHIVE_SCHEMA,
        "created_at_utc": _utc_now(),
        "workspace_root": WORKSPACE_ROOT,
        "dataset_relative_path": "data/dataset/first_five_proteins",
        "pair_count": len(pair_manifests),
        "pair_ids": [path.parent.name for path in pair_manifests],
        "file_count": len(sources),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_zip_atomically(output, sources, manifest, "colab_workspace_manifest.json")
    return output


def create_result_archive(
    bundle_dir: str | Path,
    graph_outputs_dir: str | Path,
    destination: str | Path,
) -> Path:
    """Archive one completed BioEmu bundle and its graph outputs for download."""
    bundle = Path(bundle_dir).expanduser().resolve()
    graph_outputs = Path(graph_outputs_dir).expanduser().resolve()
    output = Path(destination).expanduser().resolve()
    _require_directory(bundle, "BioEmu bundle")
    _require_directory(graph_outputs, "graph outputs")

    bundle_manifest = _read_json(bundle / "manifest.json", "BioEmu bundle manifest")
    graph_summary = _read_json(
        graph_outputs / "experiment_summary.json",
        "graph experiment summary",
    )
    for required in (graph_outputs / "path_summary.json", graph_outputs / "graph.png"):
        if not required.is_file():
            raise FileNotFoundError(f"Required graph output not found: {required}")

    graph_input = graph_summary.get("input")
    if not isinstance(graph_input, dict) or graph_input.get("mode") != "dataset_bioemu":
        raise ColabTransferError("Graph summary is not a dataset BioEmu experiment")
    protein_id = str(graph_input.get("protein_id") or "")
    pair_id = str(graph_input.get("pair_id") or "")
    if not protein_id or not pair_id:
        raise ColabTransferError("Graph summary is missing protein_id or pair_id")

    requested_samples = _positive_int(
        bundle_manifest.get("n_samples_requested"),
        "n_samples_requested",
    )
    samples = bundle_manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ColabTransferError("BioEmu bundle contains no retained samples")
    _validate_bundle_sample_paths(bundle, samples)
    bundle_sequence = bundle_manifest.get("sequence")
    if not isinstance(bundle_sequence, dict):
        raise ColabTransferError("BioEmu bundle is missing sequence metadata")
    if graph_input.get("selected_sequence_length") != bundle_sequence.get("length"):
        raise ColabTransferError("Bundle and graph sequence lengths do not match")

    sources = [
        *_archive_files(bundle, PurePosixPath("bioemu_bundle")),
        *_archive_files(graph_outputs, PurePosixPath("graph_outputs")),
    ]
    manifest = {
        "schema_version": RESULT_ARCHIVE_SCHEMA,
        "created_at_utc": _utc_now(),
        "protein_id": protein_id,
        "pair_id": pair_id,
        "requested_samples": requested_samples,
        "retained_samples": len(samples),
        "sequence": bundle_sequence,
        "bundle_root": "bioemu_bundle",
        "graph_outputs_root": "graph_outputs",
        "file_count": len(sources),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_zip_atomically(output, sources, manifest, "result_manifest.json")
    return output


def import_result_archive(
    archive_path: str | Path,
    repo_root: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Validate and import one downloaded result ZIP into canonical folders."""
    archive = Path(archive_path).expanduser().resolve()
    root = Path(repo_root).expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"Colab result archive not found: {archive}")
    _require_directory(root, "repository root")

    with zipfile.ZipFile(archive) as handle:
        _validate_zip_members(handle.infolist())
        try:
            manifest = json.loads(handle.read("result_manifest.json"))
        except KeyError as exc:
            raise ColabTransferError("Result archive is missing result_manifest.json") from exc
        except json.JSONDecodeError as exc:
            raise ColabTransferError("Result archive manifest is not valid JSON") from exc
        if manifest.get("schema_version") != RESULT_ARCHIVE_SCHEMA:
            raise ColabTransferError(
                f"Unsupported result archive schema: {manifest.get('schema_version')!r}"
            )

        protein_id = _safe_component(manifest.get("protein_id"), "protein_id")
        pair_id = _safe_component(manifest.get("pair_id"), "pair_id")
        requested_samples = _positive_int(
            manifest.get("requested_samples"),
            "requested_samples",
        )
        run_name = f"n_{requested_samples}"
        bundle_destination = (
            root
            / "data"
            / "bioemu_outputs"
            / "first_five_proteins"
            / protein_id
            / pair_id
            / run_name
        )
        graph_destination = (
            root / "outputs" / "dataset_bioemu" / protein_id / pair_id / run_name
        )
        with tempfile.TemporaryDirectory(prefix="colab_result_") as temp_name:
            temp_root = Path(temp_name)
            handle.extractall(temp_root)
            extracted_bundle = temp_root / "bioemu_bundle"
            extracted_graph = temp_root / "graph_outputs"
            _validate_extracted_result(extracted_bundle, extracted_graph, manifest)
            _install_result_directories(
                extracted_bundle,
                bundle_destination,
                extracted_graph,
                graph_destination,
                force=force,
            )

    return {
        "schema_version": RESULT_ARCHIVE_SCHEMA,
        "protein_id": protein_id,
        "pair_id": pair_id,
        "requested_samples": requested_samples,
        "retained_samples": manifest.get("retained_samples"),
        "bundle_dir": str(bundle_destination),
        "graph_outputs_dir": str(graph_destination),
    }


def _validate_extracted_result(
    bundle_dir: Path,
    graph_dir: Path,
    transfer_manifest: dict[str, Any],
) -> None:
    """Check required extracted files and cross-file identifiers."""
    bundle_manifest = _read_json(bundle_dir / "manifest.json", "extracted bundle manifest")
    graph_summary = _read_json(
        graph_dir / "experiment_summary.json",
        "extracted graph summary",
    )
    for required in (graph_dir / "path_summary.json", graph_dir / "graph.png"):
        if not required.is_file():
            raise ColabTransferError(f"Extracted result is missing {required.name}")
    samples = bundle_manifest.get("samples")
    if not isinstance(samples, list) or len(samples) != transfer_manifest.get("retained_samples"):
        raise ColabTransferError("Retained sample count does not match result manifest")
    _validate_bundle_sample_paths(bundle_dir, samples)
    graph_input = graph_summary.get("input", {})
    for key in ("protein_id", "pair_id"):
        if graph_input.get(key) != transfer_manifest.get(key):
            raise ColabTransferError(f"Extracted graph {key} does not match result manifest")


def _archive_files(source_root: Path, archive_root: PurePosixPath) -> list[tuple[Path, PurePosixPath]]:
    """Return deterministic regular-file entries rooted at ``archive_root``."""
    entries: list[tuple[Path, PurePosixPath]] = []
    for path in sorted(source_root.rglob("*")):
        relative = path.relative_to(source_root)
        if any(part in _IGNORED_PARTS for part in relative.parts):
            continue
        if path.is_symlink():
            raise ColabTransferError(f"Symlinks are not allowed in Colab archives: {path}")
        if path.is_file() and path.suffix != ".pyc":
            entries.append((path, archive_root.joinpath(*relative.parts)))
    return entries


def _validate_bundle_sample_paths(bundle_dir: Path, samples: list[Any]) -> None:
    """Require every bundle sample to reference an existing internal PDB file."""
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or not sample.get("pdb_path"):
            raise ColabTransferError(f"BioEmu sample {index} is missing pdb_path")
        relative = PurePosixPath(str(sample["pdb_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ColabTransferError(f"BioEmu sample {index} has unsafe pdb_path")
        sample_path = bundle_dir.joinpath(*relative.parts)
        if not sample_path.is_file():
            raise FileNotFoundError(f"BioEmu sample PDB not found: {sample_path}")


def _write_zip_atomically(
    destination: Path,
    sources: Iterable[tuple[Path, PurePosixPath]],
    manifest: dict[str, Any],
    manifest_name: str,
) -> None:
    """Write a deterministic-layout ZIP and replace the destination on success."""
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
    ) as temporary_handle:
        temporary = Path(temporary_handle.name)
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as handle:
            handle.writestr(manifest_name, json.dumps(manifest, indent=2) + "\n")
            for source, archive_path in sources:
                handle.write(source, archive_path.as_posix())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_zip_members(members: Iterable[zipfile.ZipInfo]) -> None:
    """Reject absolute, parent-traversing, duplicate, or symlink ZIP entries."""
    seen: set[str] = set()
    for member in members:
        path = PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ColabTransferError(f"Unsafe archive member: {member.filename!r}")
        if member.filename in seen:
            raise ColabTransferError(f"Duplicate archive member: {member.filename!r}")
        seen.add(member.filename)
        unix_mode = member.external_attr >> 16
        if (unix_mode & 0o170000) == 0o120000:
            raise ColabTransferError(f"Archive symlinks are not allowed: {member.filename!r}")


def _prepare_destination(path: Path, force: bool) -> None:
    """Require an absent destination unless explicit replacement is enabled."""
    if not path.exists():
        return
    if not force:
        raise FileExistsError(f"Destination already exists; use --force to replace it: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _install_result_directories(
    bundle_source: Path,
    bundle_destination: Path,
    graph_source: Path,
    graph_destination: Path,
    *,
    force: bool,
) -> None:
    """Install both result trees without exposing partial or lost results."""
    if not force:
        _prepare_destination(bundle_destination, force=False)
        _prepare_destination(graph_destination, force=False)
    bundle_destination.parent.mkdir(parents=True, exist_ok=True)
    graph_destination.parent.mkdir(parents=True, exist_ok=True)

    staged: list[tuple[Path, Path]] = []
    backups: list[tuple[Path, Path]] = []
    installed: list[Path] = []
    try:
        for source, destination in (
            (bundle_source, bundle_destination),
            (graph_source, graph_destination),
        ):
            stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
            shutil.rmtree(stage)
            shutil.copytree(source, stage)
            staged.append((stage, destination))

        for _, destination in staged:
            if destination.exists():
                backup = Path(tempfile.mkdtemp(prefix=f".{destination.name}.backup-", dir=destination.parent))
                shutil.rmtree(backup)
                os.replace(destination, backup)
                backups.append((backup, destination))
        for stage, destination in staged:
            os.replace(stage, destination)
            installed.append(destination)
    except Exception:
        for stage, _ in staged:
            if stage.exists():
                shutil.rmtree(stage)
        for destination in reversed(installed):
            if destination.exists():
                shutil.rmtree(destination)
        for backup, destination in reversed(backups):
            if backup.exists():
                os.replace(backup, destination)
        raise
    else:
        for backup, _ in backups:
            if backup.exists():
                shutil.rmtree(backup)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    """Read a required JSON object with a contextual error."""
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ColabTransferError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ColabTransferError(f"{label} must contain a JSON object: {path}")
    return value


def _require_directory(path: Path, label: str) -> None:
    """Raise a contextual error when a required directory is absent."""
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")


def _safe_component(value: Any, label: str) -> str:
    """Validate one manifest value used as a destination path component."""
    component = str(value or "")
    if not component or component in {".", ".."} or "/" in component or "\\" in component:
        raise ColabTransferError(f"Invalid {label}: {value!r}")
    return component


def _positive_int(value: Any, label: str) -> int:
    """Return a strictly positive integer manifest field."""
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ColabTransferError(f"{label} must be a positive integer") from exc
    if integer <= 0:
        raise ColabTransferError(f"{label} must be a positive integer")
    return integer


def _utc_now() -> str:
    """Return an ISO-formatted UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    """Return a SHA-256 digest for transfer verification and logging."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
