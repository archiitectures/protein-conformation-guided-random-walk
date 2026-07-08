"""Shortest-path search utilities for conformation graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np


@dataclass(slots=True)
class PathResult:
    """Structured shortest-path result."""

    node_ids: list[str]
    total_cost: float
    edge_scores: list[dict[str, float | str]]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
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
    """Find the Dijkstra shortest path between endpoint node IDs."""
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
