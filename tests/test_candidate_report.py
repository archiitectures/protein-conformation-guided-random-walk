"""Tests for candidate-path report and output integration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np

import guided_conformational_path.large_synthetic as experiment
from guided_conformational_path.candidate_report import write_candidate_path_reports
from guided_conformational_path.large_synthetic import LargeSyntheticConfig, LargeSyntheticRun
from guided_conformational_path.nodes import ConformationNode
from guided_conformational_path.path_search import PathResult, find_top_k_paths
from scripts.run_bioemu_precomputed_experiment import parse_args as parse_bioemu_args
from scripts.run_dataset_bioemu_experiment import parse_args as parse_dataset_args
from scripts.run_large_synthetic_experiment import parse_args as parse_synthetic_args


def _candidate_graph() -> nx.Graph:
    graph = nx.Graph()
    malicious = '<script>alert("x")</script>'
    for left, right, weight in (
        ("A", "best", 1.0),
        ("best", "B", 1.0),
        ("A", malicious, 1.2),
        (malicious, "B", 1.2),
        ("A", "other", 1.4),
        ("other", "B", 1.4),
    ):
        graph.add_edge(left, right, weight=weight, score=weight)
    return graph


def test_candidate_report_is_versioned_escaped_sortable_and_deterministic(tmp_path: Path) -> None:
    graph = _candidate_graph()
    candidates = find_top_k_paths(graph, "A", "B", k=3, scan_limit=20)

    first_json, first_html = write_candidate_path_reports(tmp_path / "first", graph, candidates)
    _, second_html = write_candidate_path_reports(tmp_path / "second", graph, candidates)

    payload = json.loads(first_json.read_text(encoding="utf-8"))
    html_text = first_html.read_text(encoding="utf-8")
    assert payload["schema_version"] == "candidate_paths_v1"
    assert payload["settings"]["k"] == 3
    assert len(payload["exact_paths"]) == 3
    assert "document.querySelectorAll('th')" in html_text
    assert "&lt;script&gt;alert" in html_text
    assert '<script>alert("x")</script>' not in html_text
    assert "Path graph thumbnail" in html_text
    assert html_text == second_html.read_text(encoding="utf-8")


def test_write_run_outputs_keeps_alternative_visuals_disabled_by_default(tmp_path: Path) -> None:
    run = _run_with_candidates()
    config = LargeSyntheticConfig(
        outputs_dir=tmp_path,
        write_animation=False,
        write_pymol=False,
        write_py3dmol=False,
    )

    experiment.write_run_outputs(run, config)

    summary = json.loads((tmp_path / "experiment_summary.json").read_text(encoding="utf-8"))
    assert (tmp_path / "candidate_paths.json").is_file()
    assert (tmp_path / "candidate_paths.html").is_file()
    assert summary["candidate_paths"]["exact_count"] == 3
    assert "alternative_path_visuals" not in summary
    assert not (tmp_path / "alternative_paths").exists()


def test_alternative_visuals_are_deduplicated_and_separate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run = _run_with_candidates()
    calls: list[tuple[str, tuple[str, ...]]] = []

    def fake_gif(output_dir, candidate_run, **kwargs):
        destination = Path(output_dir) / "path_morph.gif"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"gif")
        calls.append((str(destination.parent), tuple(candidate_run.path_result.node_ids)))
        return destination

    monkeypatch.setattr(experiment, "write_path_morph_gif", fake_gif)
    config = LargeSyntheticConfig(
        outputs_dir=tmp_path,
        write_animation=True,
        write_pymol=False,
        write_py3dmol=False,
        write_alternative_path_visuals=True,
    )

    experiment.write_run_outputs(run, config)

    expected = {
        tuple(candidate.node_ids)
        for candidate in [*run.candidate_paths.exact_paths, *run.candidate_paths.diverse_paths]
        if candidate.node_ids != run.path_result.node_ids
    }
    alternative_calls = [path for directory, path in calls if "alternative_paths" in directory]
    assert set(alternative_calls) == expected
    assert len(alternative_calls) == len(expected)
    assert all((tmp_path / "alternative_paths" / f"candidate_{candidate.global_cost_rank:03d}").is_dir()
               for candidate in run.candidate_paths.exact_paths[1:])


def test_candidate_path_cli_settings_propagate(monkeypatch) -> None:
    common = [
        "--path-top-k", "6",
        "--min-path-edge-diversity", "0.35",
        "--path-candidate-scan-limit", "80",
        "--write-alternative-path-visuals",
    ]
    for parser, required in (
        (parse_synthetic_args, []),
        (parse_bioemu_args, []),
        (parse_dataset_args, ["--pair-dir", "pair", "--bioemu-bundle-dir", "bundle"]),
    ):
        monkeypatch.setattr(sys, "argv", ["experiment.py", *required, *common])
        args = parser()
        assert args.path_top_k == 6
        assert args.min_path_edge_diversity == 0.35
        assert args.path_candidate_scan_limit == 80
        assert args.write_candidate_report is True
        assert args.write_alternative_path_visuals is True


def _run_with_candidates() -> LargeSyntheticRun:
    graph = _candidate_graph()
    coords = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]])
    endpoint_a = ConformationNode("A", "A_endpoint", coords)
    endpoint_b = ConformationNode("B", "B_endpoint", coords + 1.0)
    for node_id in graph.nodes:
        if node_id == "A":
            node = endpoint_a
        elif node_id == "B":
            node = endpoint_b
        else:
            node = ConformationNode(node_id, "external", coords + 0.5)
        graph.nodes[node_id].update(conformation=node, source=node.source, valid=True)
    candidates = find_top_k_paths(graph, "A", "B", k=3, scan_limit=20)
    best = candidates.best_path
    result = PathResult(
        node_ids=list(best.node_ids),
        total_cost=best.total_cost,
        edge_scores=[dict(edge) for edge in best.edge_scores],
        metadata=dict(best.metadata),
    )
    return LargeSyntheticRun(
        endpoint_a=endpoint_a,
        endpoint_b=endpoint_b,
        graph=graph,
        path_result=result,
        summary={},
        fallback_used=False,
        candidate_paths=candidates,
    )
