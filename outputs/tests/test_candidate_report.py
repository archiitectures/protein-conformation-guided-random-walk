"""Tests for candidate-path report and output integration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

import guided_conformational_path.large_synthetic as experiment
from guided_conformational_path.candidate_report import write_candidate_path_reports
from guided_conformational_path.large_synthetic import (
    LargeSyntheticConfig,
    LargeSyntheticRun,
    _select_random_non_top_path_results,
)
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
        "--num-top-gifs-generated", "5",
        "--num-bottom-gifs-generated", "2",
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
        assert args.num_top_gifs_generated == 5
        assert args.num_bottom_gifs_generated == 2
        assert args.write_candidate_report is True
        assert args.write_alternative_path_visuals is True


def test_write_run_outputs_generates_default_split_and_summary_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run = _run_with_ranked_candidates()
    calls: list[tuple[str, tuple[str, ...]]] = []

    def fake_gif(output_dir, candidate_run, output_filename="path_morph.gif", **kwargs):
        destination = Path(output_dir) / output_filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"gif")
        calls.append((destination.name, tuple(candidate_run.path_result.node_ids)))
        return destination

    monkeypatch.setattr(experiment, "write_path_morph_gif", fake_gif)
    config = LargeSyntheticConfig(
        outputs_dir=tmp_path,
        write_pymol=False,
        write_py3dmol=False,
        num_top_gifs_generated=6,
        num_bottom_gifs_generated=1,
    )

    experiment.write_run_outputs(run, config)

    summary = json.loads((tmp_path / "experiment_summary.json").read_text(encoding="utf-8"))
    gif_names = [name for name, _ in calls]
    assert gif_names[0] == "path_morph.gif"
    assert gif_names[1:7] == [
        "top_exact_01_global_001.gif",
        "top_exact_02_global_002.gif",
        "top_exact_03_global_003.gif",
        "top_diverse_01_global_002.gif",
        "top_diverse_02_global_003.gif",
        "top_diverse_03_global_004.gif",
    ]
    assert gif_names[7] == "random_non_top_01.gif"
    assert len(summary["categorized_path_gifs"]) == 7
    assert summary["categorized_path_gifs"][0]["relative_output_path"] == "path_gifs/top_exact_01_global_001.gif"
    assert summary["categorized_path_gifs"][0]["global_cost_rank"] == 1
    random_entry = next(item for item in summary["categorized_path_gifs"] if item["category"] == "random_non_top")
    assert random_entry["relative_output_path"] == "path_gifs/random_non_top_01.gif"
    assert random_entry["edge_scores"]
    assert random_entry["path_metadata"]["selection_category"] == "random_non_top"


def test_write_run_outputs_uses_odd_top_split(tmp_path: Path, monkeypatch) -> None:
    run = _run_with_ranked_candidates()
    calls: list[str] = []

    def fake_gif(output_dir, candidate_run, output_filename="path_morph.gif", **kwargs):
        destination = Path(output_dir) / output_filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"gif")
        calls.append(destination.name)
        return destination

    monkeypatch.setattr(experiment, "write_path_morph_gif", fake_gif)
    config = LargeSyntheticConfig(
        outputs_dir=tmp_path,
        write_pymol=False,
        write_py3dmol=False,
        num_top_gifs_generated=5,
        num_bottom_gifs_generated=0,
    )

    experiment.write_run_outputs(run, config)

    assert calls[1:] == [
        "top_exact_01_global_001.gif",
        "top_exact_02_global_002.gif",
        "top_exact_03_global_003.gif",
        "top_diverse_01_global_002.gif",
        "top_diverse_02_global_003.gif",
    ]


def test_write_run_outputs_zero_categorized_counts_disable_only_extra_groups(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run = _run_with_ranked_candidates()
    gif_dir = tmp_path / "path_gifs"
    gif_dir.mkdir()
    (gif_dir / "top_exact_09_global_009.gif").write_bytes(b"stale")
    (gif_dir / "random_non_top_09.gif").write_bytes(b"stale")
    run.summary["categorized_path_gifs"] = [{"relative_output_path": "stale"}]
    calls: list[str] = []

    def fake_gif(output_dir, candidate_run, output_filename="path_morph.gif", **kwargs):
        destination = Path(output_dir) / output_filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"gif")
        calls.append(destination.name)
        return destination

    monkeypatch.setattr(experiment, "write_path_morph_gif", fake_gif)
    config = LargeSyntheticConfig(
        outputs_dir=tmp_path,
        write_pymol=False,
        write_py3dmol=False,
        num_top_gifs_generated=0,
        num_bottom_gifs_generated=0,
    )

    experiment.write_run_outputs(run, config)

    summary = json.loads((tmp_path / "experiment_summary.json").read_text(encoding="utf-8"))
    assert calls == ["path_morph.gif"]
    assert "categorized_path_gifs" not in summary
    assert not gif_dir.exists()


def test_rerun_cleans_stale_categorized_gifs_and_preserves_unrelated_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run = _run_with_ranked_candidates()
    gif_dir = tmp_path / "path_gifs"
    gif_dir.mkdir()
    for filename in (
        "top_exact_01_global_001.gif",
        "top_exact_04_global_004.gif",
        "top_diverse_03_global_004.gif",
        "random_non_top_03.gif",
    ):
        (gif_dir / filename).write_bytes(b"stale")
    (gif_dir / "manually_curated.gif").write_bytes(b"keep")
    (gif_dir / "notes.txt").write_text("keep", encoding="utf-8")

    def fake_gif(output_dir, candidate_run, output_filename="path_morph.gif", **kwargs):
        destination = Path(output_dir) / output_filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"current")
        return destination

    monkeypatch.setattr(experiment, "write_path_morph_gif", fake_gif)
    config = LargeSyntheticConfig(
        outputs_dir=tmp_path,
        write_pymol=False,
        write_py3dmol=False,
        num_top_gifs_generated=1,
        num_bottom_gifs_generated=0,
    )

    experiment.write_run_outputs(run, config)

    assert {path.name for path in gif_dir.iterdir()} == {
        "top_exact_01_global_001.gif",
        "manually_curated.gif",
        "notes.txt",
    }
    summary = json.loads((tmp_path / "experiment_summary.json").read_text(encoding="utf-8"))
    assert [entry["relative_output_path"] for entry in summary["categorized_path_gifs"]] == [
        "path_gifs/top_exact_01_global_001.gif"
    ]


def test_write_run_outputs_uses_available_paths_without_cross_category_substitution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run = _run_with_ranked_candidates(k=2)
    calls: list[str] = []

    def fake_gif(output_dir, candidate_run, output_filename="path_morph.gif", **kwargs):
        destination = Path(output_dir) / output_filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"gif")
        calls.append(destination.name)
        return destination

    monkeypatch.setattr(experiment, "write_path_morph_gif", fake_gif)
    config = LargeSyntheticConfig(
        outputs_dir=tmp_path,
        write_pymol=False,
        write_py3dmol=False,
        num_top_gifs_generated=8,
        num_bottom_gifs_generated=3,
        path_candidate_scan_limit=5,
    )

    experiment.write_run_outputs(run, config)

    assert calls[1:] == [
        "top_exact_01_global_001.gif",
        "top_exact_02_global_002.gif",
        "top_diverse_01_global_002.gif",
        "top_diverse_02_global_003.gif",
        "random_non_top_01.gif",
        "random_non_top_02.gif",
    ]


def test_write_run_outputs_skips_all_gifs_when_animation_disabled(tmp_path: Path, monkeypatch) -> None:
    run = _run_with_ranked_candidates()
    gif_dir = tmp_path / "path_gifs"
    gif_dir.mkdir()
    (gif_dir / "top_diverse_02_global_003.gif").write_bytes(b"stale")
    (gif_dir / "random_non_top_01.gif").write_bytes(b"stale")
    run.summary["categorized_path_gifs"] = [{"relative_output_path": "stale"}]

    def fail_gif(*args, **kwargs):
        raise AssertionError("GIF writer should not be called when animation is disabled")

    monkeypatch.setattr(experiment, "write_path_morph_gif", fail_gif)
    config = LargeSyntheticConfig(
        outputs_dir=tmp_path,
        write_animation=False,
        write_pymol=False,
        write_py3dmol=False,
    )

    experiment.write_run_outputs(run, config)

    summary = json.loads((tmp_path / "experiment_summary.json").read_text(encoding="utf-8"))
    assert "path_morph_gif" not in summary
    assert "categorized_path_gifs" not in summary
    assert not gif_dir.exists()


def test_random_non_top_paths_are_selected_once_for_gifs_and_reports(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run = _run_with_ranked_candidates(k=2)
    selected = _select_random_non_top_path_results(
        run,
        count=2,
        scan_limit=8,
        seed=7,
    )
    selection_calls = 0
    rendered_random_ids: list[int] = []

    def fake_select(*args, **kwargs):
        nonlocal selection_calls
        selection_calls += 1
        return selected

    def fake_gif(output_dir, candidate_run, output_filename="path_morph.gif", **kwargs):
        destination = Path(output_dir) / output_filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"gif")
        if output_filename.startswith("random_non_top_"):
            rendered_random_ids.append(id(candidate_run.path_result))
        return destination

    monkeypatch.setattr(experiment, "_select_random_non_top_path_results", fake_select)
    monkeypatch.setattr(experiment, "write_path_morph_gif", fake_gif)
    config = LargeSyntheticConfig(
        outputs_dir=tmp_path,
        write_pymol=False,
        write_py3dmol=False,
        num_top_gifs_generated=0,
        num_bottom_gifs_generated=2,
        path_candidate_scan_limit=8,
    )

    experiment.write_run_outputs(run, config)

    payload = json.loads((tmp_path / "candidate_paths.json").read_text(encoding="utf-8"))
    report_html = (tmp_path / "candidate_paths.html").read_text(encoding="utf-8")
    random_entries = payload["random_non_top_paths"]
    assert selection_calls == 1
    assert rendered_random_ids == [id(path) for path in selected]
    assert payload["schema_version"] == "candidate_paths_v1"
    assert payload["returned_counts"]["exact"] == len(run.candidate_paths.exact_paths)
    assert payload["returned_counts"]["diverse"] == len(run.candidate_paths.diverse_paths)
    assert payload["returned_counts"]["random_non_top"] == 2
    assert [entry["node_ids"] for entry in random_entries] == [path.node_ids for path in selected]
    assert all(entry["category"] == "random_non_top" for entry in random_entries)
    assert all(entry["global_cost_rank"] is not None for entry in random_entries)
    assert all(entry["absolute_delta"] > 0 for entry in random_entries)
    assert all(0.0 <= entry["edge_jaccard"] <= 1.0 for entry in random_entries)
    assert all(0.0 <= entry["edge_diversity"] <= 1.0 for entry in random_entries)
    assert all(entry["edge_scores"] for entry in random_entries)
    assert "Random non-top comparisons" in report_html
    assert "random_non_top" in report_html
    assert "Edge overlap" in report_html
    assert report_html.count('aria-label="Path graph thumbnail"') >= len(random_entries)


def test_zero_bottom_count_adds_no_random_report_entries(tmp_path: Path, monkeypatch) -> None:
    run = _run_with_ranked_candidates()

    def fail_select(*args, **kwargs):
        raise AssertionError("random paths should not be selected when the bottom count is zero")

    def fake_gif(output_dir, candidate_run, output_filename="path_morph.gif", **kwargs):
        destination = Path(output_dir) / output_filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"gif")
        return destination

    monkeypatch.setattr(experiment, "_select_random_non_top_path_results", fail_select)
    monkeypatch.setattr(experiment, "write_path_morph_gif", fake_gif)
    config = LargeSyntheticConfig(
        outputs_dir=tmp_path,
        write_pymol=False,
        write_py3dmol=False,
        num_top_gifs_generated=2,
        num_bottom_gifs_generated=0,
    )

    experiment.write_run_outputs(run, config)

    payload = json.loads((tmp_path / "candidate_paths.json").read_text(encoding="utf-8"))
    report_html = (tmp_path / "candidate_paths.html").read_text(encoding="utf-8")
    assert "random_non_top_paths" not in payload
    assert "random_non_top" not in payload["returned_counts"]
    assert "Random non-top comparisons" not in report_html


def test_random_non_top_selection_is_deterministic_exclusive_and_bounded() -> None:
    run = _run_with_ranked_candidates()

    first = _select_random_non_top_path_results(run, count=2, scan_limit=8, seed=7)
    second = _select_random_non_top_path_results(run, count=2, scan_limit=8, seed=7)
    bounded = _select_random_non_top_path_results(run, count=2, scan_limit=4, seed=7)

    excluded = {
        tuple(candidate.node_ids)
        for candidate in [*run.candidate_paths.exact_paths, *run.candidate_paths.diverse_paths]
    }
    excluded.add(tuple(run.path_result.node_ids))
    assert [path.node_ids for path in first] == [path.node_ids for path in second]
    assert len({tuple(path.node_ids) for path in first}) == len(first)
    assert all(tuple(path.node_ids) not in excluded for path in first)
    assert bounded == []


def test_random_non_top_selection_strictly_bounds_generator_consumption(monkeypatch) -> None:
    run = _run_with_ranked_candidates()
    consumed = 0

    def counting_paths(*args, **kwargs):
        nonlocal consumed
        while True:
            consumed += 1
            yield ["A", "best", "B"]

    monkeypatch.setattr(experiment.nx, "shortest_simple_paths", counting_paths)

    selected = _select_random_non_top_path_results(run, count=1, scan_limit=4, seed=7)

    assert selected == []
    assert consumed == 4


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("num_top_gifs_generated", -1),
        ("num_bottom_gifs_generated", -2),
    ],
)
def test_write_run_outputs_rejects_negative_gif_counts_before_rendering(
    tmp_path: Path,
    monkeypatch,
    field_name: str,
    field_value: int,
) -> None:
    run = _run_with_ranked_candidates()
    config = LargeSyntheticConfig(
        outputs_dir=tmp_path,
        write_pymol=False,
        write_py3dmol=False,
    )
    setattr(config, field_name, field_value)

    def fail_render(*args, **kwargs):
        raise AssertionError("rendering should not start for invalid GIF counts")

    monkeypatch.setattr(experiment, "write_outputs", fail_render)
    monkeypatch.setattr(experiment, "write_path_morph_gif", fail_render)

    with pytest.raises(ValueError, match=rf"{field_name} must be nonnegative"):
        experiment.write_run_outputs(run, config)


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


def _ranked_candidate_graph() -> nx.Graph:
    graph = nx.Graph()
    for left, right, weight in (
        ("A", "best", 1.0),
        ("best", "B", 1.0),
        ("A", "alpha", 1.05),
        ("alpha", "B", 1.05),
        ("A", "beta", 1.10),
        ("beta", "B", 1.10),
        ("A", "gamma", 1.15),
        ("gamma", "B", 1.15),
        ("A", "delta1", 0.70),
        ("delta1", "delta2", 0.80),
        ("delta2", "B", 0.90),
        ("A", "eps1", 0.60),
        ("eps1", "eps2", 0.95),
        ("eps2", "B", 0.95),
    ):
        graph.add_edge(left, right, weight=weight, score=weight)
    return graph


def _run_with_ranked_candidates(k: int = 4) -> LargeSyntheticRun:
    graph = _ranked_candidate_graph()
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
    candidates = find_top_k_paths(graph, "A", "B", k=k, scan_limit=10)
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
        summary={"input": {"protein_name": "Glucokinase", "uniprot_id": "P35557", "protein_id": "P35557"}},
        fallback_used=False,
        candidate_paths=candidates,
    )
