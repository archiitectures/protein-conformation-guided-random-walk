"""Tests for graph construction and path search."""

from __future__ import annotations

import numpy as np

from guided_conformational_path.generators import DummyPerturbationGenerator
from guided_conformational_path.graph import build_conformation_graph
from guided_conformational_path.nodes import ConformationNode
from guided_conformational_path.path_search import find_shortest_path
from guided_conformational_path.scoring import RMSDScorer
from guided_conformational_path.visualization import plot_graph


def test_graph_construction_node_and_edge_counts() -> None:
    endpoint_a = ConformationNode("A", "A_endpoint", _coords())
    endpoint_b = ConformationNode("B", "B_endpoint", _coords() + np.array([0.0, 1.0, 0.0]))
    generator = DummyPerturbationGenerator(seed=1)
    generated_a = generator.generate_from_endpoint(endpoint_a, 2)
    generated_b = generator.generate_from_endpoint(endpoint_b, 1)

    graph = build_conformation_graph(endpoint_a, endpoint_b, generated_a, generated_b, RMSDScorer())

    assert graph.number_of_nodes() == 5
    assert graph.number_of_edges() == 10


def test_shortest_path_returns_endpoint_order() -> None:
    endpoint_a = ConformationNode("A", "A_endpoint", _coords())
    endpoint_b = ConformationNode("B", "B_endpoint", _coords() + np.array([0.0, 2.0, 0.0]))
    graph = build_conformation_graph(endpoint_a, endpoint_b, [], [], RMSDScorer())

    result = find_shortest_path(graph, "A", "B")

    assert result.node_ids[0] == "A"
    assert result.node_ids[-1] == "B"
    assert result.total_cost >= 0.0


def test_plot_graph_supports_sparse_edge_view(tmp_path) -> None:
    endpoint_a = ConformationNode("A", "A_endpoint", _coords())
    endpoint_b = ConformationNode("B", "B_endpoint", _coords() + np.array([0.0, 2.0, 0.0]))
    generated = DummyPerturbationGenerator(noise_scale=0.01, seed=1).generate_from_endpoint(endpoint_a, 2)
    graph = build_conformation_graph(endpoint_a, endpoint_b, generated, [], RMSDScorer())
    result = find_shortest_path(graph, "A", "B")
    output_path = tmp_path / "sparse_graph.png"

    plot_graph(
        graph,
        result.node_ids,
        output_path,
        max_edge_weight=0.5,
        non_path_edge_alpha=0.125,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def _coords() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [3.8, 0.0, 0.0],
            [7.6, 0.0, 0.0],
            [11.4, 0.0, 0.0],
        ]
    )
