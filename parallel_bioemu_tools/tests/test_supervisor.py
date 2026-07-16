from __future__ import annotations

from dataclasses import asdict

import pytest

from parallel_bioemu_tools.manifests import PairRunSpec, ShardAssignment, write_json
from parallel_bioemu_tools.supervisor import SupervisorConfig, run_supervised_shard

from test_worker import FakeStorage, _write_fake_bundle


def _shard_manifest(tmp_path):
    task = PairRunSpec(
        protein_id="protein_1",
        pair_id="pair_x",
        fasta_path="/tmp/input.fasta",
        dataset_pair_dir="/tmp/pair",
        n_samples=32,
        estimated_cost=5.0,
    )
    shard = ShardAssignment(
        run_id="run1",
        shard_id="shard_000",
        shard_index=0,
        total_shards=1,
        tasks=(task,),
        total_estimated_cost=5.0,
    )
    path = tmp_path / "shard_000.json"
    write_json(path, asdict(shard))
    return path


def test_supervisor_retries_until_success(monkeypatch, tmp_path) -> None:
    shard_manifest = _shard_manifest(tmp_path)
    storage = FakeStorage()
    calls = {"count": 0}

    monkeypatch.setattr(
        "parallel_bioemu_tools.supervisor.ensure_allowed_gpu",
        lambda allowed: type("Report", (), {"describe": lambda self: "", "allowed_gpu_types": tuple(allowed)})(),
    )

    def flaky_run(fasta_path, output_dir, n_samples, seed=None, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("transient failure")
        return _write_fake_bundle(output_dir, n_samples)

    monkeypatch.setattr("parallel_bioemu_tools.worker.run_bioemu_from_fasta", flaky_run)
    delays: list[float] = []
    result = run_supervised_shard(
        shard_manifest,
        storage=storage,
        config=SupervisorConfig(
            workspace_root=tmp_path / "workspace",
            max_attempts=2,
            retry_delay_seconds=5.0,
        ),
        sleep=delays.append,
    )
    assert result.attempts_used == 2
    assert len(result.completions) == 1
    assert delays == [5.0]


def test_supervisor_stops_after_max_attempts(monkeypatch, tmp_path) -> None:
    shard_manifest = _shard_manifest(tmp_path)
    storage = FakeStorage()
    monkeypatch.setattr(
        "parallel_bioemu_tools.supervisor.ensure_allowed_gpu",
        lambda allowed: type("Report", (), {"describe": lambda self: "", "allowed_gpu_types": tuple(allowed)})(),
    )
    monkeypatch.setattr(
        "parallel_bioemu_tools.worker.run_bioemu_from_fasta",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("persistent failure")),
    )
    with pytest.raises(RuntimeError, match="persistent failure"):
        run_supervised_shard(
            shard_manifest,
            storage=storage,
            config=SupervisorConfig(
                workspace_root=tmp_path / "workspace",
                max_attempts=2,
                retry_delay_seconds=0.0,
            ),
            sleep=lambda *_: None,
        )


def test_supervisor_propagates_heartbeat_and_retry_attempt(monkeypatch, tmp_path) -> None:
    shard_manifest = _shard_manifest(tmp_path)
    observed = []
    monkeypatch.setattr(
        "parallel_bioemu_tools.supervisor.ensure_allowed_gpu",
        lambda allowed: type("Report", (), {"allowed_gpu_types": tuple(allowed)})(),
    )

    def fake_worker(path, *, storage, config):
        observed.append(config)
        if len(observed) == 1:
            raise RuntimeError("retry")
        return []

    monkeypatch.setattr("parallel_bioemu_tools.supervisor.run_worker_shard", fake_worker)
    result = run_supervised_shard(
        shard_manifest,
        storage=FakeStorage(),
        config=SupervisorConfig(
            workspace_root=tmp_path / "workspace",
            max_attempts=2,
            chunk_size=20,
            worker_id="worker-a",
            heartbeat_interval_seconds=30,
            enable_heartbeat=True,
        ),
        sleep=lambda *_: None,
    )

    assert result.attempts_used == 2
    assert [config.attempt_id for config in observed] == [1, 2]
    assert all(config.enable_heartbeat for config in observed)
    assert all(config.worker_id == "worker-a" for config in observed)
    assert all(config.heartbeat_interval_seconds == 30 for config in observed)
