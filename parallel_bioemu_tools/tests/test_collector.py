from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone

import pytest

from parallel_bioemu_tools.collector import collect_run
from parallel_bioemu_tools.manifests import CollectorConfig, PairRunSpec, RunManifest, ShardAssignment, write_json


class MockStorage:
    def __init__(self, base_dir) -> None:
        self.base_dir = base_dir
        self.downloads: list[str] = []

    def download_prefix(self, remote_prefix: str, local_dir):
        self.downloads.append(remote_prefix)
        local_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = local_dir / "manifest.json"
        nodes_dir = local_dir / "nodes"
        nodes_dir.mkdir(exist_ok=True)
        (nodes_dir / "sample_000.pdb").write_text("ATOM\n", encoding="utf-8")
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "bioemu_bundle_v1",
                    "sequence": {"id": "x", "sequence": "AAA", "length": 3},
                    "n_samples_requested": 8,
                    "samples": [{"pdb_path": "nodes/sample_000.pdb"}],
                    "metadata": {},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (local_dir / "_worker_completed.json").write_text(
            json.dumps(
                {
                    "run_id": "run1",
                    "shard_id": "shard_000",
                    "protein_id": "protein_1",
                    "pair_id": "pair_x",
                    "n_samples": 8,
                    "bundle_path": "/content/bundle",
                    "remote_manifest_path": f"{remote_prefix}/manifest.json",
                    "remote_marker_path": f"{remote_prefix}/_worker_completed.json",
                    "sample_count": 1,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return local_dir

    def verify_remote_files(self, remote_paths: list[str]) -> None:
        assert remote_paths == ["run1/protein_1/pair_x/n_8/_worker_completed.json"]


def test_collector_downloads_and_runs_graph_pipeline(monkeypatch, tmp_path) -> None:
    pair = PairRunSpec(
        protein_id="protein_1",
        pair_id="pair_x",
        fasta_path="/tmp/input.fasta",
        dataset_pair_dir="/tmp/pair",
        n_samples=8,
        estimated_cost=1.0,
    )
    run_manifest = RunManifest(
        run_id="run1",
        hf_repo_id="org/repo",
        pairs=(pair,),
        shards=(ShardAssignment(run_id="run1", shard_id="shard_000", shard_index=0, total_shards=1, tasks=(pair,), total_estimated_cost=1.0),),
    )
    manifest_path = write_json(tmp_path / "run_manifest.json", asdict(run_manifest))
    storage = MockStorage(tmp_path)
    calls: list[tuple[str, str]] = []

    def fake_run_dataset_bioemu_experiment(config):
        calls.append((str(config.dataset_pair_dir), str(config.bioemu_bundle_dir)))
        return object()

    def fake_write_run_outputs(run, config):
        config.outputs_dir.mkdir(parents=True, exist_ok=True)
        (config.outputs_dir / "experiment_summary.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr("parallel_bioemu_tools.collector.run_dataset_bioemu_experiment", fake_run_dataset_bioemu_experiment)
    monkeypatch.setattr("parallel_bioemu_tools.collector.write_run_outputs", fake_write_run_outputs)
    monkeypatch.setattr("parallel_bioemu_tools.collector.load_dataset_pair_selection", lambda path: object())
    monkeypatch.setattr(
        "parallel_bioemu_tools.collector.load_adapted_bioemu_nodes",
        lambda selection, bundle_dir, max_samples=None: [object()],
    )

    results = collect_run(
        manifest_path,
        storage=storage,
        config=CollectorConfig(download_root=tmp_path / "downloads", graph_root=tmp_path / "graphs"),
    )
    assert len(results) == 1
    assert calls
    assert storage.downloads == ["run1/protein_1/pair_x/n_8"]
    assert (tmp_path / "graphs" / "run1" / "protein_1" / "pair_x" / "n_8" / "experiment_summary.json").is_file()


def test_collector_refuses_overwrite_existing_destinations(monkeypatch, tmp_path) -> None:
    pair = PairRunSpec(
        protein_id="protein_1",
        pair_id="pair_x",
        fasta_path="/tmp/input.fasta",
        dataset_pair_dir="/tmp/pair",
        n_samples=8,
        estimated_cost=1.0,
    )
    run_manifest = RunManifest(
        run_id="run1",
        hf_repo_id="org/repo",
        pairs=(pair,),
        shards=(ShardAssignment(run_id="run1", shard_id="shard_000", shard_index=0, total_shards=1, tasks=(pair,), total_estimated_cost=1.0),),
    )
    manifest_path = write_json(tmp_path / "run_manifest.json", asdict(run_manifest))
    graph_dir = tmp_path / "graphs" / "run1" / "protein_1" / "pair_x" / "n_8"
    graph_dir.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        collect_run(
            manifest_path,
            storage=MockStorage(tmp_path),
            config=CollectorConfig(download_root=tmp_path / "downloads", graph_root=tmp_path / "graphs"),
        )
