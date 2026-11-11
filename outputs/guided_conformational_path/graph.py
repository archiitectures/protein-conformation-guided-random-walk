"""NetworkX graph construction for candidate conformation paths."""

from __future__ import annotations

from itertools import combinations

import networkx as nx

from guided_conformational_path.nodes import ConformationNode
from guided_conformational_path.scoring import EdgeScorer
from guided_conformational_path.validity import check_node_validity


def build_conformation_graph(
    endpoint_a: ConformationNode,
    endpoint_b: ConformationNode,
    generated_a: list[ConformationNode],
    generated_b: list[ConformationNode],
    scorer: EdgeScorer,
    lambda_validity: float = 0.1,
    k_neighbors: int | None = None,
) -> nx.Graph:
    """Build a weighted conformation graph from endpoints and generated nodes.

    Each ``ConformationNode`` becomes one NetworkX node keyed by ``node_id``.
    Edges are scored by the supplied ``EdgeScorer`` and then augmented by a
    lightweight validity penalty:

    ``weight = scorer.score(i, j) + lambda_validity * (penalty_i + penalty_j)``

    If ``k_neighbors`` is ``None``, all node pairs are connected. Otherwise, the
    function keeps each node's ``k`` lowest-cost incident edges, producing a
    sparse undirected graph for larger candidate sets.
    """
    nodes = [endpoint_a, endpoint_b, *generated_a, *generated_b]
    graph = nx.Graph()
    validity_by_id = {node.node_id: check_node_validity(node) for node in nodes}

    for node in nodes:
        validity = validity_by_id[node.node_id]
        graph.add_node(
            node.node_id,
            conformation=node,
            source=node.source,
            validity_score=validity.score,
            valid=validity.valid,
            validity_messages=validity.messages,
        )

    candidate_edges: list[tuple[float, str, str, float]] = []
    for node_i, node_j in combinations(nodes, 2):
        raw_score = scorer.score(node_i, node_j)
        validity_penalty = validity_by_id[node_i.node_id].score + validity_by_id[node_j.node_id].score
        weight = raw_score + lambda_validity * validity_penalty
        candidate_edges.append((weight, node_i.node_id, node_j.node_id, raw_score))

    if k_neighbors is None:
        selected_edges = candidate_edges
    else:
        selected_edges = _select_knn_edges(candidate_edges, graph.nodes, k_neighbors)

    for weight, node_i, node_j, raw_score in selected_edges:
        graph.add_edge(node_i, node_j, weight=weight, score=raw_score)

    return graph


def _select_knn_edges(
    candidate_edges: list[tuple[float, str, str, float]],
    node_ids: object,
    k_neighbors: int,
) -> set[tuple[float, str, str, float]]:
    """Select undirected k-nearest-neighbor edges by edge cost.

    The candidate edge tuple is ``(weight, node_i, node_j, raw_score)``. Because
    the graph is undirected, an edge selected by either endpoint is retained.
    """
    if k_neighbors <= 0:
        raise ValueError("k_neighbors must be positive")
    selected: set[tuple[float, str, str, float]] = set()
    ids = list(node_ids)
    for node_id in ids:
        incident = [
            edge
            for edge in candidate_edges
            if edge[1] == node_id or edge[2] == node_id
        ]
        incident.sort(key=lambda edge: edge[0])
        selected.update(incident[:k_neighbors])
    return selected
