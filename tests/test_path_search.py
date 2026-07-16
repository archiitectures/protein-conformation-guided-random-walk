"""Tests for top-K conformational path search."""

from __future__ import annotations

import json
import math

import networkx as nx
import pytest

from guided_conformational_path.path_search import (
    CandidatePathCollection,
    find_candidate_paths,
    find_shortest_path,
    find_top_k_paths,
)


def graph(edges: list[tuple[str, str, float]]) -> nx.Graph:
    result = nx.Graph()
    result.add_weighted_edges_from(edges)
    for _, _, data in result.edges(data=True):
        data["score"] = data["weight"]
    return result


def test_k_one_is_shortest_path_and_ordered() -> None:
    g = graph([("A", "B", 1), ("B", "D", 2), ("A", "C", 2), ("C", "D", 2)])
    shortest = find_shortest_path(g, "A", "D")
    result = find_top_k_paths(g, "A", "D", k=1)
    assert result[0].node_ids == shortest.node_ids
    assert result[0].total_cost == shortest.total_cost
    assert [item.total_cost for item in result.exact_paths] == sorted(item.total_cost for item in result.exact_paths)


def test_paths_are_unique_and_simple_with_ties_and_fewer_than_k() -> None:
    g = graph([("A", "B", 1), ("B", "D", 1), ("A", "C", 1), ("C", "D", 1)])
    paths = find_candidate_paths(g, "A", "D", k=10)
    assert len(paths) == 2
    assert len({tuple(item.node_ids) for item in paths}) == len(paths)
    assert all(len(item.node_ids) == len(set(item.node_ids)) for item in paths)
    assert [item.global_cost_rank for item in paths] == [1, 2]


def test_disconnected_graph_preserves_networkx_behavior() -> None:
    with pytest.raises(nx.NetworkXNoPath):
        find_top_k_paths(graph([("A", "B", 1), ("C", "D", 1)]), "A", "D")


@pytest.mark.parametrize("kwargs", [{"k": 0}, {"diversity_threshold": -0.1}, {"diversity_threshold": 1.1}, {"scan_limit": 1, "k": 2}])
def test_invalid_parameters(kwargs: dict[str, float | int]) -> None:
    with pytest.raises(ValueError):
        find_top_k_paths(graph([("A", "B", 1)]), "A", "B", **kwargs)


@pytest.mark.parametrize("weight", [-1.0, math.inf, math.nan])
def test_invalid_weights(weight: float) -> None:
    with pytest.raises(ValueError):
        find_top_k_paths(graph([("A", "B", weight)]), "A", "B")


def test_diversity_scan_bound_and_metrics() -> None:
    g = graph([
        ("A", "B", 1), ("B", "D", 1), ("A", "C", 1), ("C", "D", 1),
        ("A", "E", 2), ("E", "D", 2),
    ])
    result = find_top_k_paths(g, "A", "D", k=2, diversity_threshold=0.5, scan_limit=3)
    assert len(result.exact_paths) == 2
    assert all(item.edge_diversity >= 0.5 for item in result.diverse_paths)
    assert all(item.node_ids != result.best_path.node_ids for item in result.diverse_paths)
    assert result.exact_paths[1].absolute_delta >= 0
    assert result.scan_count <= 3


def test_zero_best_cost_and_serialization() -> None:
    g = graph([("A", "B", 0), ("B", "D", 0), ("A", "C", 0), ("C", "D", 2)])
    result = find_top_k_paths(g, "A", "D", k=2)
    assert result.best_path.relative_delta == 0.0
    assert result.exact_paths[1].relative_delta is None
    payload = result.to_dict()
    assert payload["schema_version"] == "candidate_paths_v1"
    assert payload["scan_count"] >= 2
    assert payload["settings"]["k"] == 2
    assert payload["returned_counts"] == {"exact": 2, "diverse": 1}
    json.dumps(payload, allow_nan=False)
    json.dumps(result.best_path.to_dict())


def test_collection_type_and_single_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    import guided_conformational_path.path_search as path_search

    g = graph([("A", "B", 1), ("B", "D", 1), ("A", "C", 1), ("C", "D", 1)])
    original = path_search.nx.shortest_simple_paths
    calls = 0

    def counted(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(path_search.nx, "shortest_simple_paths", counted)
    result = path_search.find_top_k_paths(g, "A", "D", k=2)
    assert isinstance(result, CandidatePathCollection)
    assert calls == 1
    assert result.best_path.global_cost_rank == 1


def test_sparse_200_node_smoke() -> None:
    g = nx.Graph()
    for index in range(199):
        g.add_edge(str(index), str(index + 1), weight=1.0)
    for index in range(0, 195, 5):
        g.add_edge(str(index), str(index + 5), weight=1.0)
    result = find_top_k_paths(g, "0", "199", k=4)
    assert result.best_path.node_ids[0] == "0"
    assert result.best_path.node_ids[-1] == "199"
