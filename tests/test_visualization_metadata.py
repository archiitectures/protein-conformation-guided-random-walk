"""Tests for protein labels in GIF and HTML visualization outputs."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np

from guided_conformational_path.candidate_report import write_candidate_path_reports
from guided_conformational_path.colab_pair import run_colab_pair
from guided_conformational_path.large_synthetic import (
    LargeSyntheticRun,
    path_morph_frames,
    write_py3dmol_path_html,
)
from guided_conformational_path.nodes import ConformationNode
from guided_conformational_path.path_search import PathResult, find_top_k_paths
import guided_conformational_path.protein_metadata as protein_metadata


def test_resolver_prefers_local_name_and_skips_lookup(monkeypatch) -> None:
    summary = {
        "input": {
            "mode": "dataset_bioemu",
            "protein_name": "glucokinase",
            "uniprot_id": "P35557",
            "protein_id": "P35557",
        }
    }

    def fail_lookup(*args, **kwargs):
        raise AssertionError("UniProt lookup should not run when a local protein name is present")

    monkeypatch.setattr(protein_metadata, "_fetch_uniprot_name", fail_lookup)
    metadata = protein_metadata.resolve_protein_metadata(summary)

    assert metadata["name"] == "glucokinase"
    assert metadata["accession"] == "P35557"
    assert metadata["display_label"] == "glucokinase (P35557)"
    assert metadata["source"] == "local"


def test_resolver_uses_one_api_lookup_then_file_cache(tmp_path: Path, monkeypatch) -> None:
    cache_path = tmp_path / "uniprot_cache.json"
    monkeypatch.setattr(protein_metadata, "_UNIPROT_CACHE_PATH", cache_path)
    monkeypatch.setattr(protein_metadata, "_read_repository_uniprot_name", lambda accession: None)
    calls: list[str] = []

    def fake_lookup(accession: str, *, timeout_seconds: float) -> str | None:
        calls.append(accession)
        return "ATP-dependent enzyme"

    monkeypatch.setattr(protein_metadata, "_fetch_uniprot_name", fake_lookup)
    first = protein_metadata.resolve_protein_metadata({"input": {"uniprot_id": "P35557"}})
    second = protein_metadata.resolve_protein_metadata({"input": {"protein_id": "P35557"}})

    assert first["display_label"] == "ATP-dependent enzyme (P35557)"
    assert second["display_label"] == "ATP-dependent enzyme (P35557)"
    assert calls == ["P35557"]
    assert json.loads(cache_path.read_text(encoding="utf-8"))["P35557"] == "ATP-dependent enzyme"


def test_resolver_falls_back_cleanly_offline(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(protein_metadata, "_UNIPROT_CACHE_PATH", tmp_path / "empty_cache.json")
    monkeypatch.setattr(protein_metadata, "_read_repository_uniprot_name", lambda accession: None)
    monkeypatch.setattr(protein_metadata, "_fetch_uniprot_name", lambda *args, **kwargs: None)

    metadata = protein_metadata.resolve_protein_metadata({"input": {"uniprot_id": "P35557"}})

    assert metadata["display_label"] == "P35557"
    assert metadata["source"] == "accession_only"


def test_resolver_keeps_fetched_name_when_cache_write_fails(tmp_path: Path, monkeypatch) -> None:
    blocked_parent = tmp_path / "not_a_directory"
    blocked_parent.write_text("blocked", encoding="utf-8")
    monkeypatch.setattr(
        protein_metadata,
        "_UNIPROT_CACHE_PATH",
        blocked_parent / "uniprot_cache.json",
    )
    monkeypatch.setattr(protein_metadata, "_read_repository_uniprot_name", lambda accession: None)
    monkeypatch.setattr(
        protein_metadata,
        "_fetch_uniprot_name",
        lambda accession, *, timeout_seconds: "Fetched enzyme",
    )
    summary = {"input": {"uniprot_id": "P35557"}}

    metadata = protein_metadata.resolve_protein_metadata(summary)

    assert metadata["name"] == "Fetched enzyme"
    assert metadata["source"] == "uniprot_api"
    assert summary["protein_metadata"] == metadata


def test_resolver_prefers_repository_dataset_cache_before_network(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository_cache = tmp_path / "data" / "cache" / "api" / "uniprot"
    repository_cache.mkdir(parents=True)
    (repository_cache / "P35557.json").write_text(
        json.dumps(
            {
                "proteinDescription": {
                    "recommendedName": {"fullName": {"value": "Repository enzyme"}}
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(protein_metadata, "_UNIPROT_CACHE_PATH", tmp_path / "empty.json")

    def fail_lookup(*args, **kwargs):
        raise AssertionError("network lookup should not run when repository cache is available")

    monkeypatch.setattr(protein_metadata, "_fetch_uniprot_name", fail_lookup)

    metadata = protein_metadata.resolve_protein_metadata({"input": {"uniprot_id": "P35557"}})

    assert metadata["name"] == "Repository enzyme"
    assert metadata["source"] == "repository_cache"


def test_safe_html_link_and_escaping_for_reports_and_py3dmol(tmp_path: Path) -> None:
    run = _run_with_summary(
        {
            "input": {
                "protein_name": '<script>alert("x")</script>',
                "uniprot_id": "P35557",
            }
        }
    )

    _, report_html_path = write_candidate_path_reports(tmp_path / "report", run.graph, run.candidate_paths, summary=run.summary)
    py3dmol_path = write_py3dmol_path_html(tmp_path / "viewer", run)
    report_html = report_html_path.read_text(encoding="utf-8")
    py3dmol_html = py3dmol_path.read_text(encoding="utf-8")

    assert 'https://www.uniprot.org/uniprotkb/P35557/entry' in report_html
    assert '&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; (P35557)' in report_html
    assert '<script>alert("x")</script>' not in report_html
    assert 'https://www.uniprot.org/uniprotkb/P35557/entry' in py3dmol_html
    assert '&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; (P35557)' in py3dmol_html
    assert '<script>alert("x")</script>' not in py3dmol_html


def test_synthetic_summary_does_not_attempt_lookup(monkeypatch) -> None:
    calls: list[str] = []

    def fake_lookup(accession: str, *, timeout_seconds: float) -> str | None:
        calls.append(accession)
        return "unused"

    monkeypatch.setattr(protein_metadata, "_fetch_uniprot_name", fake_lookup)
    metadata = protein_metadata.resolve_protein_metadata({"input": {"mode": "synthetic", "n_residues": 40}})

    assert metadata["display_label"] == ""
    assert calls == []


def test_path_morph_frame_titles_include_resolved_protein_label() -> None:
    run = _run_with_summary({"input": {"protein_name": "Glucokinase", "uniprot_id": "P35557"}})

    frames = path_morph_frames(run, frames_per_edge=2)

    assert frames[0]["title"].startswith("Glucokinase (P35557) | Segment 1/")


def test_colab_pair_propagates_gif_count_settings(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    pair_dir = workspace / "data/dataset/first_five_proteins/P35557/pairs/PAIR_A__PAIR_B"
    sequence_dir = pair_dir / "sequence"
    sequence_dir.mkdir(parents=True)
    fasta_path = sequence_dir / "bioemu_selected.fasta"
    fasta_path.write_text(">P35557\nAAAA\n", encoding="utf-8")
    (pair_dir / "manifest.json").write_text(
        json.dumps(
            {
                "protein_id": "P35557",
                "pair_id": "PAIR_A__PAIR_B",
                "paths": {"bioemu_sequence": "sequence/bioemu_selected.fasta"},
            }
        ),
        encoding="utf-8",
    )
    bundle_dir = tmp_path / "content" / "runs" / "bioemu_bundles" / "P35557" / "PAIR_A__PAIR_B" / "n_5"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "manifest.json").write_text("{}", encoding="utf-8")
    archive_path = tmp_path / "content" / "exports" / "P35557__PAIR_A__PAIR_B__n_5.zip"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_bytes(b"zip")
    captured = {}

    monkeypatch.setattr("guided_conformational_path.colab_pair._is_reusable_bundle", lambda *args, **kwargs: True)
    monkeypatch.setattr("guided_conformational_path.colab_pair.write_run_outputs", lambda run, config: None)
    monkeypatch.setattr("guided_conformational_path.colab_pair.create_result_archive", lambda *args, **kwargs: None)
    monkeypatch.setattr("guided_conformational_path.colab_pair.sha256_file", lambda path: "sha")
    monkeypatch.setattr("guided_conformational_path.colab_pair.format_run_summary", lambda run, path: "summary")

    def fake_run_dataset(config):
        captured["top"] = config.num_top_gifs_generated
        captured["bottom"] = config.num_bottom_gifs_generated
        return SimpleNamespace(
            summary={"n_generated_nodes": 5},
            path_result=SimpleNamespace(node_ids=["A", "B"], total_cost=1.0),
        )

    monkeypatch.setattr("guided_conformational_path.colab_pair.run_dataset_bioemu_experiment", fake_run_dataset)

    result = run_colab_pair(
        "PAIR_A__PAIR_B",
        5,
        workspace_dir=workspace,
        content_root=tmp_path / "content",
        num_top_gifs_generated=9,
        num_bottom_gifs_generated=4,
    )

    assert captured == {"top": 9, "bottom": 4}
    assert result["archive_sha256"] == "sha"


def _run_with_summary(summary: dict[str, object]) -> LargeSyntheticRun:
    graph = nx.Graph()
    coords = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]])
    endpoint_a = ConformationNode("A", "A_endpoint", coords)
    endpoint_b = ConformationNode("B", "B_endpoint", coords + 1.0)
    for left, right, weight in (
        ("A", "mid1", 1.0),
        ("mid1", "B", 1.0),
        ("A", "mid2", 1.1),
        ("mid2", "B", 1.1),
    ):
        graph.add_edge(left, right, weight=weight, score=weight)
    for node_id in graph.nodes:
        if node_id == "A":
            node = endpoint_a
        elif node_id == "B":
            node = endpoint_b
        else:
            node = ConformationNode(node_id, "external", coords + 0.5)
        graph.nodes[node_id].update(conformation=node, source=node.source, valid=True)
    candidates = find_top_k_paths(graph, "A", "B", k=2, scan_limit=10)
    best = candidates.best_path
    return LargeSyntheticRun(
        endpoint_a=endpoint_a,
        endpoint_b=endpoint_b,
        graph=graph,
        path_result=PathResult(
            node_ids=list(best.node_ids),
            total_cost=best.total_cost,
            edge_scores=[dict(edge) for edge in best.edge_scores],
            metadata=dict(best.metadata),
        ),
        summary=summary,
        fallback_used=False,
        candidate_paths=candidates,
    )
