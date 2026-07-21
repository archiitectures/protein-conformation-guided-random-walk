from __future__ import annotations

import json
from pathlib import Path

import pytest

import parallel_bioemu_tools.vscode_batch as batch_module
from parallel_bioemu_tools.vscode_batch import (
    ColabBatchJob,
    run_vscode_colab_batch,
    verify_local_job,
)


class FakeClient:
    def __init__(self, *, fail_pair: str | None = None) -> None:
        self.fail_pair = fail_pair
        self.remote_files: set[str] = set()
        self.uploads: list[str] = []
        self.executions: list[tuple[str, bool]] = []
        self.executed_code: list[str] = []
        self.downloads: list[str] = []
        self.created_kernels = 0
        self.status_calls = 0

    def status(self) -> dict[str, object]:
        self.status_calls += 1
        return {"connections": 1}

    def upload(self, source: str | Path, remote_name: str) -> None:
        assert Path(source).is_file()
        self.uploads.append(remote_name)

    def create_kernel(self) -> str:
        self.created_kernels += 1
        return "kernel-1"

    def remote_file_exists(self, remote_path: str) -> bool:
        return remote_path in self.remote_files

    def download(self, remote_path: str, destination: str | Path) -> Path:
        output = Path(destination)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"archive")
        self.downloads.append(remote_path)
        return output

    def execute(
        self,
        kernel_id: str,
        code: str,
        label: str,
        *,
        allow_disconnect: bool = False,
        timeout_seconds: float = 8 * 60 * 60,
    ) -> str:
        del kernel_id, timeout_seconds
        self.executions.append((label, allow_disconnect))
        self.executed_code.append(code)
        if self.fail_pair and self.fail_pair in code:
            raise RuntimeError("remote failure")
        if "run_colab_pair" in code and "result =" in code:
            for protein_id, pair_id in (
                ("P00889", "1CTS_A__2CTS_A"),
                ("P35557", "1V4T_A__1V4S_A"),
            ):
                if pair_id in code:
                    self.remote_files.add(
                        f"/content/exports/{protein_id}__{pair_id}__n_200.zip"
                    )
        return ""


def test_batch_runs_sequentially_and_releases_after_both_verify(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "workspace.zip"
    archive.write_bytes(b"workspace")
    jobs = _jobs()
    client = FakeClient()
    monkeypatch.setattr(
        batch_module,
        "_import_archive",
        lambda _archive, root, job, _label: _write_complete_output(root, job),
    )

    results = run_vscode_colab_batch(
        client,
        jobs,
        repo_root=tmp_path,
        workspace_archive=archive,
        required_gpu="L4",
        release_on_success=True,
    )

    assert [result.protein_id for result in results] == ["P00889", "P35557"]
    assert client.uploads == ["/content/guided_conformation_workspace.zip"]
    assert client.created_kernels == 1
    assert client.downloads == [job.remote_archive_path for job in jobs]
    pair_labels = [label for label, _ in client.executions if label.startswith("BioEmu")]
    assert pair_labels == [
        "BioEmu and graph: P00889 / 1CTS_A__2CTS_A",
        "BioEmu and graph: P35557 / 1V4T_A__1V4S_A",
    ]
    assert client.executions[-1] == ("Release Colab runtime", True)


def test_batch_failure_preserves_first_result_and_does_not_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "workspace.zip"
    archive.write_bytes(b"workspace")
    jobs = _jobs()
    client = FakeClient(fail_pair="1V4T_A__1V4S_A")
    monkeypatch.setattr(
        batch_module,
        "_import_archive",
        lambda _archive, root, job, _label: _write_complete_output(root, job),
    )

    with pytest.raises(RuntimeError, match="remote failure"):
        run_vscode_colab_batch(
            client,
            jobs,
            repo_root=tmp_path,
            workspace_archive=archive,
            release_on_success=True,
        )

    assert verify_local_job(tmp_path, jobs[0]) is not None
    assert verify_local_job(tmp_path, jobs[1]) is None
    assert all(label != "Release Colab runtime" for label, _ in client.executions)


def test_batch_recovers_remote_archive_without_resampling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "workspace.zip"
    archive.write_bytes(b"workspace")
    job = _jobs()[0]
    client = FakeClient()
    client.remote_files.add(job.remote_archive_path)
    monkeypatch.setattr(
        batch_module,
        "_import_archive",
        lambda _archive, root, imported_job, _label: _write_complete_output(root, imported_job),
    )

    run_vscode_colab_batch(
        client,
        [job],
        repo_root=tmp_path,
        workspace_archive=archive,
    )

    assert client.downloads == [job.remote_archive_path]
    assert all(not label.startswith("BioEmu and graph") for label, _ in client.executions)


def test_new_archive_does_not_require_immediate_contents_api_visibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "workspace.zip"
    archive.write_bytes(b"workspace")
    job = _jobs()[0]
    client = FakeClient()
    client.remote_file_exists = lambda remote_path: False
    monkeypatch.setattr(
        batch_module,
        "_import_archive",
        lambda _archive, root, imported_job, _label: _write_complete_output(root, imported_job),
    )

    results = run_vscode_colab_batch(
        client,
        [job],
        repo_root=tmp_path,
        workspace_archive=archive,
    )

    assert results[0].protein_id == job.protein_id
    assert client.downloads == [job.remote_archive_path]
    assert any(label.startswith("Validate remote archive") for label, _ in client.executions)


def test_verified_local_jobs_skip_remote_setup(tmp_path: Path) -> None:
    archive = tmp_path / "workspace.zip"
    archive.write_bytes(b"workspace")
    job = _jobs()[0]
    _write_complete_output(tmp_path, job)
    client = FakeClient()

    results = run_vscode_colab_batch(
        client,
        [job],
        repo_root=tmp_path,
        workspace_archive=archive,
    )

    assert results[0].source == "existing_local_output"
    assert client.uploads == []
    assert client.created_kernels == 0


def test_batch_uses_the_requested_dataset_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "workspace.zip"
    archive.write_bytes(b"workspace")
    job = ColabBatchJob("Q13257", "1DUJ_A__1S2H_A", 200)
    client = FakeClient()
    dataset_label = "non_ideal_transitions_n3"
    monkeypatch.setattr(
        batch_module,
        "_import_archive",
        lambda _archive, root, imported_job, label: _write_complete_output(
            root, imported_job, dataset_label=label
        ),
    )

    run_vscode_colab_batch(
        client,
        [job],
        repo_root=tmp_path,
        workspace_archive=archive,
        required_gpu="T4",
        dataset_label=dataset_label,
    )

    assert client.uploads == ["/content/non_ideal_transitions_n3_workspace.zip"]
    assert any(
        'dataset_label="non_ideal_transitions_n3"' in code
        for code in client.executed_code
    )
    assert verify_local_job(tmp_path, job, dataset_label=dataset_label) is not None


def test_batch_resumes_an_existing_kernel_without_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "workspace.zip"
    archive.write_bytes(b"workspace")
    job = _jobs()[0]
    client = FakeClient()
    monkeypatch.setattr(
        batch_module,
        "_import_archive",
        lambda _archive, root, imported_job, _label: _write_complete_output(root, imported_job),
    )

    run_vscode_colab_batch(
        client,
        [job],
        repo_root=tmp_path,
        workspace_archive=archive,
        recover_kernel_id="known-kernel",
    )

    assert client.uploads == []
    assert client.created_kernels == 0
    assert client.executions[0][0].startswith("BioEmu and graph")


def _jobs() -> list[ColabBatchJob]:
    return [
        ColabBatchJob("P00889", "1CTS_A__2CTS_A", 200),
        ColabBatchJob("P35557", "1V4T_A__1V4S_A", 200),
    ]


def _write_complete_output(
    root: Path,
    job: ColabBatchJob,
    *,
    dataset_label: str = "first_five_proteins",
) -> None:
    bundle = (
        root
        / "data/bioemu_outputs"
        / dataset_label
        / job.protein_id
        / job.pair_id
        / job.run_name
    )
    graph_root = root / "outputs/dataset_bioemu"
    if dataset_label != "first_five_proteins":
        graph_root = graph_root / dataset_label
    graph = graph_root / job.protein_id / job.pair_id / job.run_name
    bundle.mkdir(parents=True, exist_ok=True)
    graph.mkdir(parents=True, exist_ok=True)
    nodes_dir = bundle / "nodes"
    nodes_dir.mkdir()
    (nodes_dir / "sample.pdb").write_text("END\n", encoding="ascii")
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "n_samples_requested": job.n_samples,
                "samples": [{"pdb_path": "nodes/sample.pdb"}] * 180,
            }
        ),
        encoding="utf-8",
    )
    (graph / "experiment_summary.json").write_text(
        json.dumps(
            {
                "input": {"protein_id": job.protein_id, "pair_id": job.pair_id},
                "n_generated_nodes": 180,
                "n_nodes": 182,
                "n_edges": 1402,
                "path_node_ids": ["A", "sample", "B"],
                "total_cost": 2.5,
            }
        ),
        encoding="utf-8",
    )
    (graph / "path_summary.json").write_text(
        json.dumps({"node_ids": ["A", "sample", "B"], "total_cost": 2.5}),
        encoding="utf-8",
    )
    (graph / "graph.png").write_bytes(b"png")
