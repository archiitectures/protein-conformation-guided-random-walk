"""Tests for no-Drive Colab workspace and pair-result archives."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

from guided_conformational_path.colab_transfer import (
    ColabTransferError,
    RESULT_ARCHIVE_SCHEMA,
    WORKSPACE_ARCHIVE_SCHEMA,
    create_result_archive,
    create_workspace_archive,
    import_result_archive,
)
from scripts.colab_transfer import parse_args


def test_create_workspace_archive_contains_source_and_five_pairs(tmp_path: Path) -> None:
    repo_root, dataset_root = _fake_workspace(tmp_path)
    output = tmp_path / "transfer" / "workspace.zip"

    archive = create_workspace_archive(repo_root, dataset_root, output)

    with zipfile.ZipFile(archive) as handle:
        names = set(handle.namelist())
        manifest = json.loads(handle.read("colab_workspace_manifest.json"))
    assert manifest["schema_version"] == WORKSPACE_ARCHIVE_SCHEMA
    assert manifest["pair_count"] == 5
    assert "protein-conformation-guided-random-walk/pyproject.toml" in names
    assert (
        "protein-conformation-guided-random-walk/"
        "guided_conformational_path/dataset_bioemu_adapter.py"
    ) in names
    assert sum(name.endswith("/manifest.json") for name in names) == 5
    assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)


def test_workspace_archive_supports_non_first_five_dataset_labels(tmp_path: Path) -> None:
    repo_root, _ = _fake_workspace(tmp_path)
    dataset_root = repo_root / "data/dataset/non_ideal_transitions_n3"
    for index in range(3):
        pair_dir = dataset_root / f"P{index}" / "pairs" / f"A{index}__B{index}"
        pair_dir.mkdir(parents=True)
        (pair_dir / "manifest.json").write_text("{}", encoding="utf-8")

    archive = create_workspace_archive(repo_root, dataset_root, tmp_path / "workspace.zip")

    with zipfile.ZipFile(archive) as handle:
        manifest = json.loads(handle.read("colab_workspace_manifest.json"))
        names = set(handle.namelist())
    assert manifest["dataset_label"] == "non_ideal_transitions_n3"
    assert manifest["dataset_relative_path"] == "data/dataset/non_ideal_transitions_n3"
    assert manifest["pair_count"] == 3
    assert any("data/dataset/non_ideal_transitions_n3/" in name for name in names)


def test_create_and_import_result_archive_round_trip(tmp_path: Path) -> None:
    bundle_dir, graph_dir = _fake_result(tmp_path)
    archive = create_result_archive(bundle_dir, graph_dir, tmp_path / "PTEST_A__B_n_2.zip")
    local_repo = tmp_path / "local_repo"
    local_repo.mkdir()

    result = import_result_archive(archive, local_repo)

    expected_bundle = (
        local_repo
        / "data/bioemu_outputs/first_five_proteins/PTEST/A__B/n_2"
    )
    expected_graph = local_repo / "outputs/dataset_bioemu/PTEST/A__B/n_2"
    assert result["schema_version"] == RESULT_ARCHIVE_SCHEMA
    assert result["retained_samples"] == 2
    assert Path(result["bundle_dir"]) == expected_bundle
    assert Path(result["graph_outputs_dir"]) == expected_graph
    assert (expected_bundle / "manifest.json").exists()
    assert (expected_bundle / "nodes/bioemu_000001.pdb").exists()
    assert (expected_graph / "experiment_summary.json").exists()
    assert (expected_graph / "graph.png").exists()


def test_import_result_archive_uses_declared_dataset_label(tmp_path: Path) -> None:
    bundle_dir, graph_dir = _fake_result(tmp_path)
    archive = create_result_archive(
        bundle_dir,
        graph_dir,
        tmp_path / "result.zip",
        dataset_label="non_ideal_transitions_n3",
    )
    local_repo = tmp_path / "local_repo"
    local_repo.mkdir()

    result = import_result_archive(archive, local_repo)

    assert result["dataset_label"] == "non_ideal_transitions_n3"
    assert Path(result["bundle_dir"]) == (
        local_repo / "data/bioemu_outputs/non_ideal_transitions_n3/PTEST/A__B/n_2"
    )
    assert Path(result["graph_outputs_dir"]) == (
        local_repo / "outputs/dataset_bioemu/non_ideal_transitions_n3/PTEST/A__B/n_2"
    )


def test_import_result_archive_refuses_existing_destination(tmp_path: Path) -> None:
    bundle_dir, graph_dir = _fake_result(tmp_path)
    archive = create_result_archive(bundle_dir, graph_dir, tmp_path / "result.zip")
    local_repo = tmp_path / "local_repo"
    local_repo.mkdir()
    import_result_archive(archive, local_repo)

    with pytest.raises(FileExistsError, match="--force"):
        import_result_archive(archive, local_repo)


def test_force_import_preserves_existing_results_when_install_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_dir, graph_dir = _fake_result(tmp_path)
    archive = create_result_archive(bundle_dir, graph_dir, tmp_path / "result.zip")
    local_repo = tmp_path / "local_repo"
    local_repo.mkdir()
    import_result_archive(archive, local_repo)
    expected_bundle = local_repo / "data/bioemu_outputs/first_five_proteins/PTEST/A__B/n_2"
    expected_graph = local_repo / "outputs/dataset_bioemu/PTEST/A__B/n_2"
    (expected_bundle / "sentinel.txt").write_text("old bundle\n", encoding="utf-8")
    (expected_graph / "sentinel.txt").write_text("old graph\n", encoding="utf-8")

    import guided_conformational_path.colab_transfer as transfer

    original_copytree = transfer.shutil.copytree
    calls = 0

    def fail_on_second_copy(source, destination, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk full")
        return original_copytree(source, destination, *args, **kwargs)

    monkeypatch.setattr(transfer.shutil, "copytree", fail_on_second_copy)
    with pytest.raises(OSError, match="disk full"):
        import_result_archive(archive, local_repo, force=True)

    assert (expected_bundle / "sentinel.txt").read_text(encoding="utf-8") == "old bundle\n"
    assert (expected_graph / "sentinel.txt").read_text(encoding="utf-8") == "old graph\n"


def test_force_import_rolls_back_when_final_install_move_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_dir, graph_dir = _fake_result(tmp_path)
    archive = create_result_archive(bundle_dir, graph_dir, tmp_path / "result.zip")
    local_repo = tmp_path / "local_repo"
    local_repo.mkdir()
    import_result_archive(archive, local_repo)
    expected_bundle = local_repo / "data/bioemu_outputs/first_five_proteins/PTEST/A__B/n_2"
    expected_graph = local_repo / "outputs/dataset_bioemu/PTEST/A__B/n_2"
    (expected_bundle / "sentinel.txt").write_text("old bundle\n", encoding="utf-8")
    (expected_graph / "sentinel.txt").write_text("old graph\n", encoding="utf-8")

    import guided_conformational_path.colab_transfer as transfer

    original_replace = transfer.os.replace
    calls = 0

    def fail_on_final_install(source, destination):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("final move failed")
        return original_replace(source, destination)

    monkeypatch.setattr(transfer.os, "replace", fail_on_final_install)
    with pytest.raises(OSError, match="final move failed"):
        import_result_archive(archive, local_repo, force=True)

    assert (expected_bundle / "sentinel.txt").read_text(encoding="utf-8") == "old bundle\n"
    assert (expected_graph / "sentinel.txt").read_text(encoding="utf-8") == "old graph\n"


def test_import_result_archive_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("result_manifest.json", json.dumps({"schema_version": RESULT_ARCHIVE_SCHEMA}))
        handle.writestr("../escape.txt", "unsafe")
    local_repo = tmp_path / "local_repo"
    local_repo.mkdir()

    with pytest.raises(ColabTransferError, match="Unsafe archive member"):
        import_result_archive(archive, local_repo)


@pytest.mark.parametrize(
    ("member", "field", "value", "message"),
    [
        ("bioemu_bundle/manifest.json", "n_samples_requested", 3, "Requested sample count"),
        ("bioemu_bundle/manifest.json", "schema_version", "unknown", "unsupported schema"),
        ("graph_outputs/experiment_summary.json", "n_generated_nodes", 1, "generated-node count"),
        ("graph_outputs/experiment_summary.json", "n_nodes", 99, "graph node count"),
        ("graph_outputs/experiment_summary.json", "input.mode", "bioemu_precomputed", "not a dataset"),
        ("graph_outputs/experiment_summary.json", "input.selected_sequence_length", 3, "sequence lengths"),
    ],
)
def test_import_rejects_mismatched_embedded_provenance(
    tmp_path: Path,
    member: str,
    field: str,
    value: object,
    message: str,
) -> None:
    bundle_dir, graph_dir = _fake_result(tmp_path)
    archive = create_result_archive(bundle_dir, graph_dir, tmp_path / "result.zip")
    _rewrite_archive_json(archive, member, field, value)
    local_repo = tmp_path / "local_repo"
    local_repo.mkdir()

    with pytest.raises(ColabTransferError, match=message):
        import_result_archive(archive, local_repo)


def test_create_result_archive_rejects_missing_sample_pdb(tmp_path: Path) -> None:
    bundle_dir, graph_dir = _fake_result(tmp_path)
    (bundle_dir / "nodes/bioemu_000001.pdb").unlink()

    with pytest.raises(FileNotFoundError, match="sample PDB"):
        create_result_archive(bundle_dir, graph_dir, tmp_path / "result.zip")


def test_result_archive_accepts_declared_candidate_report(tmp_path: Path) -> None:
    bundle_dir, graph_dir = _fake_result(tmp_path)
    summary_path = graph_dir / "experiment_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["candidate_paths"] = {
        "schema_version": "candidate_paths_v1",
        "json": "candidate_paths.json",
        "html": "candidate_paths.html",
        "exact_count": 1,
        "diverse_count": 0,
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    (graph_dir / "candidate_paths.json").write_text(
        json.dumps({"schema_version": "candidate_paths_v1"}),
        encoding="utf-8",
    )
    (graph_dir / "candidate_paths.html").write_text("<!doctype html>", encoding="utf-8")

    archive = create_result_archive(bundle_dir, graph_dir, tmp_path / "result.zip")

    with zipfile.ZipFile(archive) as handle:
        assert "graph_outputs/candidate_paths.json" in handle.namelist()
        assert "graph_outputs/candidate_paths.html" in handle.namelist()


def test_result_archive_rejects_missing_declared_candidate_report(tmp_path: Path) -> None:
    bundle_dir, graph_dir = _fake_result(tmp_path)
    summary_path = graph_dir / "experiment_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["candidate_paths"] = {
        "schema_version": "candidate_paths_v1",
        "json": "candidate_paths.json",
        "html": "candidate_paths.html",
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ColabTransferError, match="report files are missing"):
        create_result_archive(bundle_dir, graph_dir, tmp_path / "result.zip")


def test_colab_transfer_cli_parses_commands(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["colab_transfer.py", "package-workspace", "--output", "workspace.zip"],
    )
    package_args = parse_args()
    assert package_args.command == "package-workspace"
    assert package_args.output == Path("workspace.zip")

    monkeypatch.setattr(
        sys,
        "argv",
        ["colab_transfer.py", "import-result", "pair.zip", "--force"],
    )
    import_args = parse_args()
    assert import_args.command == "import-result"
    assert import_args.archive == Path("pair.zip")
    assert import_args.force is True


def _fake_workspace(tmp_path: Path) -> tuple[Path, Path]:
    repo_root = tmp_path / "repo"
    for directory in ("guided_conformational_path", "protein_conformer_dataset", "scripts"):
        path = repo_root / directory
        path.mkdir(parents=True)
        (path / "__init__.py").write_text("", encoding="utf-8")
    (repo_root / "guided_conformational_path/dataset_bioemu_adapter.py").write_text(
        "ADAPTER = True\n",
        encoding="utf-8",
    )
    (repo_root / "guided_conformational_path/__pycache__").mkdir()
    (repo_root / "guided_conformational_path/__pycache__/ignored.pyc").write_bytes(b"ignored")
    (repo_root / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    (repo_root / "README.md").write_text("# Test\n", encoding="utf-8")

    dataset_root = repo_root / "data/dataset/first_five_proteins"
    for index in range(5):
        pair_dir = dataset_root / f"P{index}" / "pairs" / f"A{index}__B{index}"
        pair_dir.mkdir(parents=True)
        (pair_dir / "manifest.json").write_text(
            json.dumps({"protein_id": f"P{index}", "pair_id": f"A{index}__B{index}"}),
            encoding="utf-8",
        )
    return repo_root, dataset_root


def _fake_result(tmp_path: Path) -> tuple[Path, Path]:
    bundle_dir = tmp_path / "bundle"
    nodes_dir = bundle_dir / "nodes"
    nodes_dir.mkdir(parents=True)
    samples = []
    for index in range(2):
        node_path = nodes_dir / f"bioemu_{index:06d}.pdb"
        node_path.write_text("END\n", encoding="utf-8")
        samples.append(
            {
                "node_id": f"bioemu_{index:06d}",
                "pdb_path": f"nodes/{node_path.name}",
            }
        )
    (bundle_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "bioemu_bundle_v1",
                "n_samples_requested": 2,
                "sequence": {"id": "PTEST", "sequence": "AAAA", "length": 4},
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )

    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    (graph_dir / "experiment_summary.json").write_text(
        json.dumps(
            {
                "input": {
                    "mode": "dataset_bioemu",
                    "protein_id": "PTEST",
                    "pair_id": "A__B",
                    "selected_sequence_length": 4,
                },
                "n_generated_nodes": 2,
                "n_nodes": 4,
            }
        ),
        encoding="utf-8",
    )
    (graph_dir / "path_summary.json").write_text("{}", encoding="utf-8")
    (graph_dir / "graph.png").write_bytes(b"fake-png")
    return bundle_dir, graph_dir


def _rewrite_archive_json(
    archive: Path,
    member_name: str,
    field: str,
    value: object,
) -> None:
    """Replace one top-level JSON field in a test ZIP member."""
    replacement = archive.with_suffix(".replacement.zip")
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(replacement, "w") as destination:
        for member in source.infolist():
            content = source.read(member.filename)
            if member.filename == member_name:
                payload = json.loads(content)
                target = payload
                path = field.split(".")
                for component in path[:-1]:
                    target = target[component]
                target[path[-1]] = value
                content = json.dumps(payload).encode("utf-8")
            destination.writestr(member, content)
    replacement.replace(archive)
