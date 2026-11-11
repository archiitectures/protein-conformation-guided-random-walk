"""Tests for the large synthetic experiment helpers."""

from __future__ import annotations

import json

import numpy as np
import networkx as nx
from PIL import Image

from guided_conformational_path.nodes import ConformationNode
from guided_conformational_path.large_synthetic import (
    build_and_search,
    build_route_constrained_graph,
    generate_gaussian_nodes,
    generate_recursive_gaussian_nodes,
    LargeSyntheticConfig,
    load_pdb_endpoints,
    display_py3dmol_path,
    make_fallback_nodes,
    make_synthetic_endpoints,
    match_pdb_ca_records,
    PDBEndpointConfig,
    path_intermediate_count,
    path_node_centers,
    path_node_coordinate_payloads,
    path_node_marker_positions,
    path_node_pdb_blocks,
    route_family_costs,
    run_large_synthetic_experiment,
    run_pdb_endpoint_experiment,
    select_recursive_frontiers,
    SquaredRMSDScorer,
    LargeSyntheticRun,
    path_morph_frames,
    write_path_morph_gif,
    write_py3dmol_path_html,
    write_pymol_path_view,
    write_run_outputs,
)
from guided_conformational_path.io import CARecord, read_pdb_ca_records, write_pdb_ca
from guided_conformational_path.path_search import PathResult
from guided_conformational_path.validity import check_node_validity
from scripts.run_large_synthetic_experiment import parse_args


def _payload_translation(payload: dict[str, object]) -> np.ndarray:
    translation = payload["display_translation"]
    assert isinstance(translation, dict)
    return np.array([translation["x"], translation["y"], translation["z"]], dtype=float)


def _assert_minimum_payload_separation(payloads: list[dict[str, object]]) -> None:
    translations = [_payload_translation(payload) for payload in payloads]
    min_center_distance = float(payloads[0]["minimum_center_distance"])
    for idx, translation_i in enumerate(translations):
        for translation_j in translations[idx + 1 :]:
            distance = float(np.linalg.norm(translation_i - translation_j))
            assert distance >= min_center_distance - 1e-8


def _make_simple_morph_run() -> LargeSyntheticRun:
    endpoint_a = ConformationNode(
        "A",
        "A_endpoint",
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        ),
    )
    endpoint_b = ConformationNode(
        "B",
        "B_endpoint",
        np.array(
            [
                [0.0, 2.0, 0.0],
                [1.0, 2.0, 0.0],
                [2.0, 2.0, 0.0],
            ]
        ),
    )
    graph = nx.Graph()
    for node in (endpoint_a, endpoint_b):
        graph.add_node(
            node.node_id,
            conformation=node,
            source=node.source,
            validity_score=1.0,
            valid=True,
            validity_messages=[],
        )
    graph.add_edge("A", "B", weight=1.0, score=1.0)
    path_result = PathResult(
        node_ids=["A", "B"],
        total_cost=1.0,
        edge_scores=[{"from": "A", "to": "B", "weight": 1.0, "score": 1.0}],
        metadata={"n_path_nodes": 2, "n_path_edges": 1, "beta": None},
    )
    return LargeSyntheticRun(
        endpoint_a=endpoint_a,
        endpoint_b=endpoint_b,
        graph=graph,
        path_result=path_result,
        summary={},
        fallback_used=False,
    )


def _make_weighted_morph_run() -> LargeSyntheticRun:
    endpoint_a = ConformationNode(
        "A",
        "A_endpoint",
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        ),
    )
    midpoint = ConformationNode(
        "M",
        "external",
        np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
            ]
        ),
    )
    endpoint_b = ConformationNode(
        "B",
        "B_endpoint",
        np.array(
            [
                [0.0, 4.0, 0.0],
                [1.0, 4.0, 0.0],
                [2.0, 4.0, 0.0],
            ]
        ),
    )
    graph = nx.Graph()
    for node in (endpoint_a, midpoint, endpoint_b):
        graph.add_node(
            node.node_id,
            conformation=node,
            source=node.source,
            validity_score=1.0,
            valid=True,
            validity_messages=[],
        )
    graph.add_edge("A", "M", weight=1.0, score=1.0)
    graph.add_edge("M", "B", weight=1.0, score=1.0)
    path_result = PathResult(
        node_ids=["A", "M", "B"],
        total_cost=2.0,
        edge_scores=[
            {"from": "A", "to": "M", "weight": 1.0, "score": 1.0},
            {"from": "M", "to": "B", "weight": 1.0, "score": 1.0},
        ],
        metadata={"n_path_nodes": 3, "n_path_edges": 2, "beta": None},
    )
    return LargeSyntheticRun(
        endpoint_a=endpoint_a,
        endpoint_b=endpoint_b,
        graph=graph,
        path_result=path_result,
        summary={},
        fallback_used=False,
    )


def test_make_synthetic_endpoints_default_to_200_residues() -> None:
    endpoint_a, endpoint_b = make_synthetic_endpoints()

    assert endpoint_a.coords.shape == (200, 3)
    assert endpoint_b.coords.shape == (200, 3)
    assert check_node_validity(endpoint_a).valid
    assert check_node_validity(endpoint_b).valid


def test_make_synthetic_endpoints_are_globular() -> None:
    endpoint_a, _ = make_synthetic_endpoints(200)
    spread = endpoint_a.coords.std(axis=0)

    assert spread.min() > 5.0
    assert spread.max() / spread.min() < 4.0


def test_generate_gaussian_nodes_records_summary() -> None:
    endpoint_a, endpoint_b = make_synthetic_endpoints(40)

    nodes, summaries = generate_gaussian_nodes(
        [endpoint_a, endpoint_b],
        n_samples_total=6,
        noise_scale=0.02,
        seed=1,
    )

    assert len(summaries) == 2
    assert sum(summary["requested"] for summary in summaries) == 6
    assert len(nodes) <= 6
    assert all("rejected" in summary for summary in summaries)


def test_recursive_gaussian_generation_expands_generated_nodes() -> None:
    endpoint_a, endpoint_b = make_synthetic_endpoints(40)

    nodes, summaries = generate_recursive_gaussian_nodes(
        endpoint_a,
        endpoint_b,
        n_samples_total=10,
        noise_scale=0.01,
        seed=5,
        max_iterations=3,
        top_k=2,
        samples_per_parent=2,
    )

    assert len(nodes) <= 10
    assert len({node.node_id for node in nodes}) == len(nodes)
    assert {node.metadata["recursive_side"] for node in nodes} == {"A", "B"}
    assert any(str(node.metadata["parent"]).startswith(("A_rec_", "B_rec_")) for node in nodes)
    assert summaries[-1]["phase"] == "recursive_gaussian_summary"
    assert summaries[-1]["returned_total"] == len(nodes)


def test_recursive_gaussian_generation_reports_iteration_progress() -> None:
    endpoint_a, endpoint_b = make_synthetic_endpoints(40)
    events: list[dict[str, object]] = []

    nodes, _ = generate_recursive_gaussian_nodes(
        endpoint_a,
        endpoint_b,
        n_samples_total=10,
        noise_scale=0.01,
        seed=5,
        max_iterations=3,
        top_k=2,
        samples_per_parent=2,
        progress_callback=events.append,
    )

    assert events
    assert [event["iteration"] for event in events] == sorted(
        event["iteration"] for event in events
    )
    assert events[-1]["phase"] == "recursive_gaussian_iteration"
    assert events[-1]["returned_total"] == len(nodes)
    assert events[-1]["requested_total"] == 10
    assert events[-1]["frontier_a"] >= 1
    assert events[-1]["frontier_b"] >= 1


def test_recursive_frontier_selection_uses_closest_cross_side_pairs() -> None:
    endpoint_a, endpoint_b = make_synthetic_endpoints(20)
    coords = endpoint_a.coords
    a_close = ConformationNode("A_close", "A_generated", coords)
    b_close_coords = coords.copy()
    b_close_coords[0] += np.array([0.2, 0.0, 0.0])
    b_close = ConformationNode("B_close", "B_generated", b_close_coords)
    a_far_coords = coords.copy()
    a_far_coords[0] += np.array([8.0, 0.0, 0.0])
    a_far = ConformationNode("A_far", "A_generated", a_far_coords)
    b_far_coords = endpoint_b.coords.copy()
    b_far_coords[0] += np.array([8.0, 0.0, 0.0])
    b_far = ConformationNode("B_far", "B_generated", b_far_coords)

    frontiers = select_recursive_frontiers(
        [a_far, a_close],
        [b_far, b_close],
        endpoint_a,
        endpoint_b,
        SquaredRMSDScorer(),
        top_k=1,
    )

    assert [node.node_id for node in frontiers["A"]] == ["A_close"]
    assert [node.node_id for node in frontiers["B"]] == ["B_close"]


def test_fallback_path_starts_and_ends_at_endpoints() -> None:
    endpoint_a, endpoint_b = make_synthetic_endpoints(40)
    nodes, _ = make_fallback_nodes(
        endpoint_a,
        endpoint_b,
        n_waypoints=4,
        samples_per_waypoint=1,
        noise_scale=0.01,
        seed=2,
    )

    _, result = build_and_search(endpoint_a, endpoint_b, nodes, k_neighbors=None)

    assert result.node_ids[0] == "A"
    assert result.node_ids[-1] == "B"
    assert path_intermediate_count(result) >= 1


def test_route_constrained_fallback_has_four_intermediates() -> None:
    endpoint_a, endpoint_b = make_synthetic_endpoints(40)
    nodes, _ = make_fallback_nodes(
        endpoint_a,
        endpoint_b,
        n_waypoints=4,
        samples_per_waypoint=0,
        noise_scale=0.01,
        seed=2,
    )

    graph, result = build_route_constrained_graph(endpoint_a, endpoint_b, nodes)

    assert path_intermediate_count(result) == 4
    assert result.node_ids[1].startswith("low_energy")
    assert set(route_family_costs(graph)) == {"low_energy", "medium_detour", "high_detour"}


def test_run_large_synthetic_experiment_returns_summary() -> None:
    config = LargeSyntheticConfig(n_residues=40, n_samples=8)
    run = run_large_synthetic_experiment(config)

    assert run.endpoint_a.coords.shape == (40, 3)
    assert run.path_result.node_ids[0] == "A"
    assert run.path_result.node_ids[-1] == "B"
    assert "recursive_gaussian" in run.summary


def test_run_large_synthetic_experiment_can_disable_fallback() -> None:
    config = LargeSyntheticConfig(
        n_residues=40,
        n_samples=8,
        target_min_intermediates=20,
        target_max_intermediates=30,
        allow_fallback=False,
    )

    run = run_large_synthetic_experiment(config)

    assert run.fallback_used is False
    assert run.summary["fallback_allowed"] is False
    assert run.summary["recursive_path_missed_target"] is True
    assert run.graph.number_of_nodes() == run.summary["recursive_gaussian"]["n_nodes"]
    assert run.path_result.node_ids[0] == "A"
    assert run.path_result.node_ids[-1] == "B"


def test_run_large_synthetic_experiment_uses_fallback_when_allowed() -> None:
    config = LargeSyntheticConfig(
        n_residues=40,
        n_samples=8,
        target_min_intermediates=20,
        target_max_intermediates=30,
        allow_fallback=True,
    )

    run = run_large_synthetic_experiment(config)

    assert run.fallback_used is True
    assert run.summary["fallback_allowed"] is True
    assert run.summary["recursive_path_missed_target"] is True
    assert route_family_costs(run.graph)


def test_write_pymol_path_view_outputs_pdbs_and_script(tmp_path) -> None:
    config = LargeSyntheticConfig(n_residues=40, n_samples=8, outputs_dir=tmp_path)
    run = run_large_synthetic_experiment(config)

    pml_path = write_pymol_path_view(tmp_path, run)

    assert pml_path.exists()
    assert "load 00_A.pdb" in pml_path.read_text(encoding="utf-8")
    assert (tmp_path / "pymol_path" / "00_A.pdb").exists()
    assert (tmp_path / "pymol_path" / f"{len(run.path_result.node_ids) - 1:02d}_B.pdb").exists()


def test_path_morph_frames_linearly_interpolate_path_nodes() -> None:
    run = _make_simple_morph_run()

    frames = path_morph_frames(run, frames_per_edge=3)

    assert len(frames) == 3
    assert [frame["start_node_id"] for frame in frames] == ["A", "A", "A"]
    assert [frame["end_node_id"] for frame in frames] == ["B", "B", "B"]
    assert np.allclose(frames[0]["coords"], run.graph.nodes["A"]["conformation"].coords)
    assert np.allclose(frames[-1]["coords"], run.graph.nodes["B"]["conformation"].coords)
    assert np.allclose(
        frames[1]["coords"],
        0.5 * (
            run.graph.nodes["A"]["conformation"].coords
            + run.graph.nodes["B"]["conformation"].coords
        ),
    )


def test_path_morph_frames_expose_weighted_progress_and_color_metadata() -> None:
    run = _make_weighted_morph_run()

    frames = path_morph_frames(run, frames_per_edge=2)

    assert len(frames) == 4
    assert frames[0]["progress"] == 0.0
    assert frames[-1]["progress"] == 1.0
    assert frames[0]["color"] == "#1f77b4"
    assert frames[-1]["color"] == "#d62728"
    assert [frame["global_frame_index"] for frame in frames] == [0, 1, 2, 3]
    assert [frame["global_frame_count"] for frame in frames] == [4, 4, 4, 4]
    assert frames[1]["duration_weight"] < frames[2]["duration_weight"]
    assert frames[2]["progress"] < (2.0 / 3.0)


def test_write_path_morph_gif_outputs_non_empty_gif(tmp_path) -> None:
    run = _make_simple_morph_run()

    gif_path = write_path_morph_gif(tmp_path, run, frames_per_edge=3, fps=5, dpi=72)

    assert gif_path.exists()
    assert gif_path.stat().st_size > 0
    with Image.open(gif_path) as gif:
        assert getattr(gif, "n_frames", 1) >= 2


def test_write_path_morph_gif_uses_movement_weighted_frame_durations(tmp_path) -> None:
    run = _make_weighted_morph_run()

    gif_path = write_path_morph_gif(
        tmp_path,
        run,
        frames_per_edge=2,
        fps=10,
        dpi=72,
        min_duration_weight=1.0,
    )

    durations_ms: list[int] = []
    with Image.open(gif_path) as gif:
        for frame_idx in range(getattr(gif, "n_frames", 1)):
            gif.seek(frame_idx)
            durations_ms.append(int(gif.info["duration"]))

    assert durations_ms == [100, 100, 150, 150]


def test_write_run_outputs_records_path_morph_gif(tmp_path) -> None:
    config = LargeSyntheticConfig(
        n_residues=40,
        n_samples=8,
        outputs_dir=tmp_path,
        write_pymol=False,
        write_py3dmol=False,
    )
    run = run_large_synthetic_experiment(config)

    write_run_outputs(run, config)

    summary_path = tmp_path / "experiment_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert "path_morph_gif" in summary
    assert summary["path_morph_gif"] == str(tmp_path / "path_morph.gif")
    assert (tmp_path / "path_morph.gif").exists()


def test_write_run_outputs_can_disable_animation(tmp_path) -> None:
    config = LargeSyntheticConfig(
        n_residues=40,
        n_samples=8,
        outputs_dir=tmp_path,
        write_animation=False,
        write_pymol=False,
        write_py3dmol=False,
    )
    run = run_large_synthetic_experiment(config)

    write_run_outputs(run, config)

    summary_path = tmp_path / "experiment_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert "path_morph_gif" not in summary
    assert not (tmp_path / "path_morph.gif").exists()


def test_large_synthetic_cli_exposes_animation_flags(monkeypatch) -> None:
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_large_synthetic_experiment.py",
            "--no-fallback",
            "--no-animation",
            "--animation-frames-per-edge",
            "7",
            "--animation-fps",
            "12",
            "--animation-dpi",
            "144",
        ],
    )

    args = parse_args()

    assert args.allow_fallback is False
    assert args.write_animation is False
    assert args.animation_frames_per_edge == 7
    assert args.animation_fps == 12
    assert args.animation_dpi == 144


def test_path_node_pdb_blocks_cover_selected_path() -> None:
    config = LargeSyntheticConfig(n_residues=40, n_samples=8)
    run = run_large_synthetic_experiment(config)

    blocks = path_node_pdb_blocks(run)

    assert [node_id for node_id, _ in blocks] == run.path_result.node_ids
    assert all("ATOM" in pdb_text for _, pdb_text in blocks)


def test_path_node_centers_cover_selected_path() -> None:
    config = LargeSyntheticConfig(n_residues=40, n_samples=8)
    run = run_large_synthetic_experiment(config)

    centers = path_node_centers(run)

    assert [node_id for node_id, _ in centers] == run.path_result.node_ids
    assert all(len(center) == 3 for _, center in centers)


def test_path_node_marker_positions_cover_selected_path() -> None:
    config = LargeSyntheticConfig(n_residues=40, n_samples=8)
    run = run_large_synthetic_experiment(config)

    markers = path_node_marker_positions(run)

    assert [node_id for node_id, _ in markers] == run.path_result.node_ids
    assert all(len(position) == 3 for _, position in markers)
    assert len({position for _, position in markers}) == len(markers)


def test_path_node_coordinate_payloads_cover_all_residues() -> None:
    config = LargeSyntheticConfig(n_residues=40, n_samples=8)
    run = run_large_synthetic_experiment(config)

    payloads = path_node_coordinate_payloads(run)
    translations = [payload["display_translation"]["x"] for payload in payloads]

    assert [payload["node_id"] for payload in payloads] == run.path_result.node_ids
    assert all(len(payload["coords"]) == 40 for payload in payloads)
    assert all({"x", "y", "z"} <= set(payload["coords"][0]) for payload in payloads)
    assert all(float(payload["display_radius"]) > 0.0 for payload in payloads)
    assert all(float(payload["minimum_center_distance"]) > 0.0 for payload in payloads)
    if len(payloads) <= 4:
        assert translations == sorted(translations)
        assert len(set(translations)) == len(translations)
        assert {payload["layout_mode"] for payload in payloads} == {"linear"}
        _assert_minimum_payload_separation(payloads)


def test_path_node_coordinate_payloads_separate_linear_layout() -> None:
    config = LargeSyntheticConfig(
        n_residues=40,
        n_samples=0,
        target_min_intermediates=2,
        target_max_intermediates=2,
        fallback_waypoints=2,
    )
    run = run_large_synthetic_experiment(config)

    payloads = path_node_coordinate_payloads(run)

    assert len(payloads) <= 4
    assert {payload["layout_mode"] for payload in payloads} == {"linear"}
    _assert_minimum_payload_separation(payloads)


def test_path_node_coordinate_payloads_use_circular_layout_for_long_paths() -> None:
    config = LargeSyntheticConfig(
        n_residues=40,
        n_samples=0,
        target_min_intermediates=4,
        target_max_intermediates=4,
        fallback_waypoints=4,
    )
    run = run_large_synthetic_experiment(config)

    payloads = path_node_coordinate_payloads(run)
    translations = [tuple(_payload_translation(payload)) for payload in payloads]
    endpoint_distance = float(np.linalg.norm(np.array(translations[0]) - np.array(translations[-1])))
    radii = [float(np.linalg.norm(np.array(translation[:2]))) for translation in translations]
    endpoint_y = translations[0][1]
    intermediate_ys = [translation[1] for translation in translations[1:-1]]

    assert len(payloads) > 4
    assert {payload["layout_mode"] for payload in payloads} == {"circular"}
    assert endpoint_distance >= float(payloads[0]["minimum_center_distance"]) - 1e-8
    assert endpoint_distance < max(radii)
    assert all(y_coord < endpoint_y for y_coord in intermediate_ys)
    assert len(set(translations)) == len(translations)
    _assert_minimum_payload_separation(payloads)


def test_path_node_coordinate_payloads_can_disable_separation() -> None:
    config = LargeSyntheticConfig(n_residues=40, n_samples=8)
    run = run_large_synthetic_experiment(config)

    payloads = path_node_coordinate_payloads(run, separate_conformations=False)

    for payload, node_id in zip(payloads, run.path_result.node_ids):
        node = run.graph.nodes[node_id]["conformation"]
        payload_coords = np.array(
            [[coord["x"], coord["y"], coord["z"]] for coord in payload["coords"]],
            dtype=float,
        )
        assert np.allclose(payload_coords, node.coords)
        assert np.allclose(_payload_translation(payload), np.zeros(3))


def test_write_py3dmol_path_html_outputs_integrated_view(tmp_path) -> None:
    config = LargeSyntheticConfig(n_residues=40, n_samples=8, outputs_dir=tmp_path)
    run = run_large_synthetic_experiment(config)

    html_path = write_py3dmol_path_html(tmp_path, run)
    html = html_path.read_text(encoding="utf-8")

    assert html_path.exists()
    assert "3Dmol-min.js" in html
    assert "Selected A-to-B Path Structure View" in html
    assert "A ->" in html
    assert 'id="representation"' in html
    assert 'value="spheres"' in html
    assert 'value="sticks_nodes"' in html
    assert 'id="trace-density"' in html
    assert 'value="none"' in html
    assert 'id="viewer-region"' in html
    assert "#controls { position: relative; z-index: 20;" in html
    assert "#viewer { position: relative; z-index: 1;" in html
    assert "viewer.addSphere" in html
    assert "viewer.addCylinder" in html
    assert "3Dmol.js did not load" in html
    assert "separated full conformations" in html
    assert "layoutMode" in html
    assert "arranged around a circle" in html
    assert "Motion traces" in html


def test_display_py3dmol_path_returns_view_when_dependency_installed() -> None:
    config = LargeSyntheticConfig(n_residues=40, n_samples=8)
    run = run_large_synthetic_experiment(config)

    view = display_py3dmol_path(run, width=300, height=250)

    assert view is not None


def test_load_pdb_endpoints_reads_matching_ca_files(tmp_path) -> None:
    endpoint_a, endpoint_b = make_synthetic_endpoints(20)
    path_a = tmp_path / "A.pdb"
    path_b = tmp_path / "B.pdb"
    write_pdb_ca(path_a, endpoint_a.coords)
    write_pdb_ca(path_b, endpoint_b.coords)

    loaded_a, loaded_b = load_pdb_endpoints(path_a, path_b)

    assert loaded_a.coords.shape == (20, 3)
    assert loaded_b.coords.shape == (20, 3)
    assert loaded_a.file_path == path_a
    assert loaded_b.file_path == path_b
    assert loaded_a.metadata["matching"]["method_used"] == "exact"


def test_read_pdb_ca_records_parses_residue_metadata(tmp_path) -> None:
    path = tmp_path / "records.pdb"
    path.write_text(
        "ATOM      1  CA  GLY B  42A      1.000   2.000   3.000  1.00  0.00           C\n"
        "END\n",
        encoding="utf-8",
    )

    records = read_pdb_ca_records(path)

    assert len(records) == 1
    assert records[0].chain_id == "B"
    assert records[0].residue_number == 42
    assert records[0].insertion_code == "A"
    assert records[0].residue_name == "GLY"


def test_exact_pdb_matching_reorders_shuffled_records() -> None:
    records_a = [
        CARecord("A", 1, "", "ALA", np.array([0.0, 0.0, 0.0])),
        CARecord("A", 2, "", "ALA", np.array([3.8, 0.0, 0.0])),
        CARecord("A", 3, "", "ALA", np.array([7.6, 0.0, 0.0])),
    ]
    records_b = [
        CARecord("A", 3, "", "ALA", np.array([7.6, 1.0, 0.0])),
        CARecord("A", 1, "", "ALA", np.array([0.0, 1.0, 0.0])),
        CARecord("A", 2, "", "ALA", np.array([3.8, 1.0, 0.0])),
    ]

    coords_a, coords_b, metadata = match_pdb_ca_records(records_a, records_b, matching_method="exact")

    assert metadata["method_used"] == "exact"
    assert coords_a[:, 0].tolist() == [0.0, 3.8, 7.6]
    assert coords_b[:, 0].tolist() == [0.0, 3.8, 7.6]


def test_geometric_pdb_matching_handles_mismatched_residue_ids() -> None:
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [3.8, 0.0, 0.0],
            [7.6, 0.4, 0.0],
            [11.2, 1.0, 0.2],
            [14.7, 1.5, 0.4],
            [18.0, 1.6, 0.8],
            [21.4, 1.2, 1.0],
            [24.8, 0.6, 1.2],
        ]
    )
    records_a = [CARecord("A", idx + 1, "", "ALA", coord) for idx, coord in enumerate(coords)]
    records_b = [
        CARecord("B", idx + 101, "", "ALA", coord + np.array([10.0, -5.0, 2.0]))
        for idx, coord in enumerate(coords)
    ]

    coords_a, coords_b, metadata = match_pdb_ca_records(
        records_a,
        records_b,
        matching_method="exact_then_geometric",
        geometric_match_distance_cutoff=0.5,
        min_geometric_matches=8,
    )

    assert metadata["method_used"] == "geometric"
    assert coords_a.shape == coords_b.shape == (8, 3)


def test_run_pdb_endpoint_experiment_uses_pdb_input_mode(tmp_path) -> None:
    endpoint_a, endpoint_b = make_synthetic_endpoints(30)
    path_a = tmp_path / "A.pdb"
    path_b = tmp_path / "B.pdb"
    write_pdb_ca(path_a, endpoint_a.coords)
    write_pdb_ca(path_b, endpoint_b.coords)
    config = PDBEndpointConfig(
        endpoint_a_path=path_a,
        endpoint_b_path=path_b,
        n_samples=8,
        outputs_dir=tmp_path / "outputs",
        write_pymol=False,
        write_py3dmol=False,
    )

    run = run_pdb_endpoint_experiment(config)

    assert run.endpoint_a.n_residues == 30
    assert run.path_result.node_ids[0] == "A"
    assert run.path_result.node_ids[-1] == "B"
    assert run.summary["input"]["mode"] == "pdb"
    assert "matching" in run.summary["input"]
