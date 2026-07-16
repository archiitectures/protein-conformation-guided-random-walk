from __future__ import annotations

import importlib
import json

import pytest
import numpy as np

from guided_conformational_path.bioemu_io import load_bioemu_bundle
from guided_conformational_path.io import write_pdb_ca
from parallel_bioemu_tools.manifests import PairRunSpec


def api(name: str):
    package = importlib.import_module("parallel_bioemu_tools")
    if hasattr(package, name):
        return getattr(package, name)
    for module_name in ("chunking", "recovery", "watchdog", "merge", "collector"):
        try:
            module = importlib.import_module(f"parallel_bioemu_tools.{module_name}")
        except ModuleNotFoundError:
            continue
        if hasattr(module, name):
            return getattr(module, name)
    raise AssertionError(f"expected public API is missing: {name}")


def task(n_samples: int = 45, seed: int | None = 17) -> PairRunSpec:
    return PairRunSpec("P1", "pair", "/tmp/pair.fasta", "/tmp/pair", n_samples, 1.0, seed=seed)


def bundle_manifest(sequence: str, samples: list[dict]) -> dict:
    return {
        "schema_version": "bioemu_bundle_v1",
        "sequence": {"id": "protein", "sequence": sequence, "length": len(sequence)},
        "n_samples_requested": len(samples),
        "samples": samples,
    }


def test_expand_45_samples_and_seed_determinism():
    expand = api("expand_task_chunks")
    chunks = list(expand(task(), chunk_size=20))
    assert [getattr(chunk, "n_samples") for chunk in chunks] == [20, 20, 5]
    assert len({getattr(chunk, "seed") for chunk in chunks}) == 3
    assert [getattr(chunk, "seed") for chunk in chunks] == [getattr(chunk, "seed") for chunk in expand(task(), chunk_size=20)]


def test_legacy_unchunked_task_is_compatible():
    expand = api("expand_task_chunks")
    chunks = list(expand(task(8), chunk_size=None))
    assert len(chunks) == 1
    assert getattr(chunks[0], "n_samples") == 8
    assert getattr(chunks[0], "seed") == 17


def test_resume_skips_completed_chunks_and_regenerates_only_missing(tmp_path, monkeypatch):
    expand = api("expand_task_chunks")
    resume = api("resume_task_chunks")
    chunks = list(expand(task(), chunk_size=20))
    completed = {getattr(chunks[0], "chunk_id"), getattr(chunks[1], "chunk_id")}
    pending = list(resume(chunks, completed_chunk_ids=completed))
    assert pending == [chunks[2]]


def test_merge_rejects_sequence_paths_and_duplicate_ids(tmp_path):
    merge = api("merge_chunk_bundles")
    bundles = [tmp_path / "a", tmp_path / "b", tmp_path / "c"]
    for bundle in bundles[:2]:
        bundle.mkdir()
        (bundle / "manifest.json").write_text(json.dumps(bundle_manifest("AAAA", [])), encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        merge(bundles[:2], tmp_path / "out")

    (bundles[0] / "sample_0.pdb").write_text("MODEL\nENDMDL\n", encoding="utf-8")
    (bundles[0] / "sample_1.pdb").write_text("MODEL\nENDMDL\n", encoding="utf-8")
    (bundles[1] / "sample_0.pdb").write_text("MODEL\nENDMDL\n", encoding="utf-8")
    (bundles[0] / "manifest.json").write_text(json.dumps(bundle_manifest("AAAA", [
        {"node_id": "same", "pdb_path": "sample_0.pdb"},
        {"node_id": "same", "pdb_path": "sample_1.pdb"},
    ])), encoding="utf-8")
    (bundles[1] / "manifest.json").write_text(json.dumps(bundle_manifest("AAAA", [
        {"node_id": "same", "pdb_path": "sample_0.pdb"},
    ])), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate|identifier"):
        merge(bundles[:2], tmp_path / "duplicate-out")


def test_duplicate_source_ids_across_chunks_are_renamed(tmp_path):
    merge = api("merge_chunk_bundles")
    chunks = []
    for index in range(2):
        bundle = tmp_path / f"chunk-{index}"
        bundle.mkdir()
        (bundle / "sample.pdb").write_text("MODEL\nENDMDL\n", encoding="utf-8")
        (bundle / "manifest.json").write_text(json.dumps(bundle_manifest("AAAA", [
            {"node_id": "same", "pdb_path": "sample.pdb"},
        ])), encoding="utf-8")
        chunks.append(bundle)
    merged = merge(chunks, tmp_path / "merged")
    identifiers = [sample["node_id"] for sample in json.loads((merged / "manifest.json").read_text())["samples"]]
    assert len(identifiers) == len(set(identifiers))


def test_merge_rejects_mismatched_sequence_and_missing_sample_path(tmp_path):
    merge = api("merge_chunk_bundles")
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "sample.pdb").write_text("MODEL\nENDMDL\n", encoding="utf-8")
    (first / "manifest.json").write_text(json.dumps(bundle_manifest("AAAA", [{"node_id": "a", "pdb_path": "sample.pdb"}])), encoding="utf-8")
    (second / "manifest.json").write_text(json.dumps(bundle_manifest("BBBB", [{"node_id": "b", "pdb_path": "missing.pdb"}])), encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError), match="sequence|missing|PDB|sample"):
        merge([first, second], tmp_path / "out")


def test_merged_standard_bundle_loads_at_existing_io_boundary(tmp_path):
    merge = api("merge_chunk_bundles")
    chunks = []
    for index in range(2):
        bundle = tmp_path / f"chunk-{index}"
        bundle.mkdir()
        sample = bundle / "nodes" / "sample.pdb"
        write_pdb_ca(sample, np.asarray([[0.0 + index, 0.0, 0.0], [1.0 + index, 0.0, 0.0]]))
        (bundle / "manifest.json").write_text(
            json.dumps(bundle_manifest("AA", [{"node_id": f"source-{index}", "pdb_path": "nodes/sample.pdb"}])),
            encoding="utf-8",
        )
        chunks.append(bundle)
    merged = merge(chunks, tmp_path / "merged")
    assert len(load_bioemu_bundle(merged)) == 2
    assert callable(api("load_adapted_bioemu_nodes"))


def test_expected_heartbeat_and_recovery_records_are_public():
    assert api("WorkerHeartbeat") is not None
    assert api("RecoveryRecord") is not None


def test_completion_marker_and_graph_collection_are_idempotent():
    marker = api("ChunkCompletionMarker")
    assert marker is not None
    collect = api("collect_run")
    assert callable(collect)
