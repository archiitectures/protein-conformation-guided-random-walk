from __future__ import annotations

import importlib.util
import json
import sys
from types import SimpleNamespace


def load_script(name: str):
    path = f"parallel_bioemu_tools/scripts/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_worker_cli_resilient_defaults(monkeypatch):
    module = load_script("run_worker")
    monkeypatch.setattr(sys, "argv", ["run_worker", "--shard-manifest", "s.json", "--workspace-root", "w", "--hf-repo-id", "org/r"])
    args = module.parse_args()
    assert args.chunk_size == 20
    assert args.worker_id is None
    assert args.heartbeat_interval_seconds == 120


def test_supervisor_cli_resilient_defaults(monkeypatch):
    module = load_script("supervise_worker")
    monkeypatch.setattr(sys, "argv", ["supervise_worker", "--shard-manifest", "s.json", "--workspace-root", "w", "--hf-repo-id", "org/r"])
    args = module.parse_args()
    assert args.chunk_size == 20
    assert args.worker_id is None
    assert args.heartbeat_interval_seconds == 120


def test_worker_cli_enables_heartbeat(monkeypatch, tmp_path):
    module = load_script("run_worker")
    manifest = tmp_path / "shard.json"
    manifest.write_text(
        json.dumps({
            "run_id": "run1",
            "shard_id": "shard_000",
            "shard_index": 0,
            "total_shards": 1,
            "tasks": [],
            "total_estimated_cost": 0.0,
        }),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        shard_manifest=manifest,
        workspace_root=tmp_path / "worker",
        hf_repo_id="org/repo",
        overwrite=False,
        chunk_size=20,
        worker_id=None,
        heartbeat_interval_seconds=60.0,
        release_colab_runtime=False,
        allowed_gpu_types=None,
    )
    observed = {}
    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(
        module,
        "ensure_allowed_gpu",
        lambda allowed: SimpleNamespace(describe=lambda: "GPU: T4"),
    )
    monkeypatch.setattr(module, "HuggingFaceStorage", lambda repo_id: object())
    monkeypatch.setattr(
        module,
        "run_worker_shard",
        lambda path, *, storage, config: observed.setdefault("config", config) and [],
    )

    module.main()

    config = observed["config"]
    assert config.enable_heartbeat is True
    assert config.worker_id == "shard_000"
    assert config.heartbeat_interval_seconds == 60.0


def test_supervisor_cli_enables_heartbeat(monkeypatch, tmp_path):
    module = load_script("supervise_worker")
    manifest = tmp_path / "shard.json"
    manifest.write_text(json.dumps({"shard_id": "shard_000"}), encoding="utf-8")
    args = SimpleNamespace(
        shard_manifest=manifest,
        workspace_root=tmp_path / "worker",
        hf_repo_id="org/repo",
        allowed_gpu_types=None,
        overwrite=False,
        chunk_size=20,
        worker_id=None,
        heartbeat_interval_seconds=60.0,
        max_attempts=3,
        retry_delay_seconds=30.0,
        release_colab_runtime=False,
    )
    observed = {}
    result = SimpleNamespace(
        gpu_report=SimpleNamespace(describe=lambda: "GPU: T4"),
        attempts_used=1,
        completions=(),
    )
    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "HuggingFaceStorage", lambda repo_id: object())
    monkeypatch.setattr(
        module,
        "run_supervised_shard",
        lambda path, *, storage, config: observed.setdefault("config", config) and result,
    )

    module.main()

    config = observed["config"]
    assert config.enable_heartbeat is True
    assert config.worker_id == "shard_000"
    assert config.heartbeat_interval_seconds == 60.0
