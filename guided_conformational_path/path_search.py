"""Shortest-path search utilities for conformation graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np


@dataclass(slots=True)
class PathResult:
    """Structured shortest-path result.

    Attributes:
        node_ids: Ordered graph node IDs from start to end.
        total_cost: Sum of edge weights along the selected path.
        edge_scores: Per-edge raw scores, final weights, and optional probability
            diagnostics.
        metadata: Small JSON-serializable summary of path settings.
    """

    node_ids: list[str]
    total_cost: float
    edge_scores: list[dict[str, float | str]]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary for demo output."""
        return {
            "node_ids": self.node_ids,
            "total_cost": self.total_cost,
            "edge_scores": self.edge_scores,
            "metadata": self.metadata,
        }


def find_shortest_path(
    graph: nx.Graph,
    start_id: str,
    end_id: str,
    beta: float | None = None,
) -> PathResult:
    """Find the Dijkstra shortest path between endpoint node IDs.

    NetworkX's Dijkstra implementation minimizes the sum of stored edge
    ``weight`` values. The optional ``beta`` parameter does not change that
    objective; it only adds the diagnostic quantity ``exp(-beta * weight)`` to
    each returned edge record. That diagnostic can be interpreted as an
    unnormalized transition preference where lower-cost edges receive larger
    values.
    """
    node_ids = nx.shortest_path(graph, start_id, end_id, weight="weight")
    total_cost = float(nx.shortest_path_length(graph, start_id, end_id, weight="weight"))
    edge_scores: list[dict[str, float | str]] = []

    for node_i, node_j in zip(node_ids[:-1], node_ids[1:]):
        edge_data = graph[node_i][node_j]
        record: dict[str, float | str] = {
            "from": node_i,
            "to": node_j,
            "weight": float(edge_data["weight"]),
            "score": float(edge_data["score"]),
        }
        if beta is not None:
            record["unnormalized_probability"] = float(np.exp(-beta * edge_data["weight"]))
        edge_scores.append(record)

    return PathResult(
        node_ids=node_ids,
        total_cost=total_cost,
        edge_scores=edge_scores,
        metadata={
            "n_path_nodes": len(node_ids),
            "n_path_edges": max(0, len(node_ids) - 1),
            "beta": beta,
        },
    )
