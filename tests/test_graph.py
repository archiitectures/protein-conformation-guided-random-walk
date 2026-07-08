"""Tests for graph construction and path search."""

from __future__ import annotations

import numpy as np

from guided_conformational_path.generators import DummyPerturbationGenerator
from guided_conformational_path.graph import build_conformation_graph
from guided_conformational_path.nodes import ConformationNode
from guided_conformational_path.path_search import find_shortest_path
from guided_conformational_path.scoring import RMSDScorer


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


def _coords() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [3.8, 0.0, 0.0],
            [7.6, 0.0, 0.0],
            [11.4, 0.0, 0.0],
        ]
    )
