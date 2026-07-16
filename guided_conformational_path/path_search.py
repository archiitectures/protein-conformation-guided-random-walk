"""Shortest-path search utilities for conformation graphs."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal

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


@dataclass(slots=True)
class PathCandidate:
    """A ranked, serializable candidate route."""

    node_ids: list[str]
    total_cost: float
    absolute_delta: float
    relative_delta: float | None
    edge_jaccard: float
    edge_diversity: float
    intermediate_node_jaccard: float
    global_cost_rank: int
    edge_scores: list[dict[str, float | str]]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return one candidate as a JSON-compatible dictionary."""
        return {
            "node_ids": list(self.node_ids),
            "total_cost": self.total_cost,
            "absolute_delta": self.absolute_delta,
            "relative_delta": self.relative_delta,
            "edge_jaccard": self.edge_jaccard,
            "edge_diversity": self.edge_diversity,
            "intermediate_node_jaccard": self.intermediate_node_jaccard,
            "global_cost_rank": self.global_cost_rank,
            "edge_scores": [dict(edge) for edge in self.edge_scores],
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class CandidatePathCollection(Sequence[PathCandidate]):
    """Exact and diversity-filtered candidate routes."""

    best_path: PathCandidate
    exact_paths: list[PathCandidate]
    diverse_paths: list[PathCandidate]
    settings: dict[str, Any]
    scan_count: int

    def __len__(self) -> int:
        """Return the number of exact ranked paths."""
        return len(self.exact_paths)

    def __getitem__(self, index: int) -> PathCandidate:
        """Return an exact path by rank-relative index."""
        return self.exact_paths[index]

    def __iter__(self) -> Iterator[PathCandidate]:
        """Iterate over exact paths in rank order."""
        return iter(self.exact_paths)

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned candidate report payload."""
        return {
            "schema_version": "candidate_paths_v1",
            "best_path": self.best_path.to_dict(),
            "exact_paths": [path.to_dict() for path in self.exact_paths],
            "diverse_paths": [path.to_dict() for path in self.diverse_paths],
            "settings": dict(self.settings),
            "scan_count": self.scan_count,
            "returned_counts": {
                "exact": len(self.exact_paths),
                "diverse": len(self.diverse_paths),
            },
        }


TopKPathsResult = CandidatePathCollection


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


def _validate_search_parameters(k: int, diversity_threshold: float, scan_limit: int) -> None:
    if k < 1:
        raise ValueError("k must be at least 1")
    if not 0 <= diversity_threshold <= 1:
        raise ValueError("diversity_threshold must be between 0 and 1")
    if scan_limit < k:
        raise ValueError("scan_limit must be at least k")


def _validate_weights(graph: nx.Graph) -> None:
    for _, _, data in graph.edges(data=True):
        weight = data.get("weight")
        if weight is None or not np.isfinite(weight) or weight < 0:
            raise ValueError("all graph edge weights must be finite and nonnegative")


def _edge_key(graph: nx.Graph, left: str, right: str) -> tuple[str, str]:
    if graph.is_directed():
        return left, right
    return tuple(sorted((left, right)))


def _jaccard(left: set[Any], right: set[Any]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _candidate(
    graph: nx.Graph,
    node_ids: list[str],
    best_nodes: list[str],
    best_cost: float,
    rank: int,
    beta: float | None,
) -> PathCandidate:
    edge_scores: list[dict[str, float | str]] = []
    edges: set[tuple[str, str]] = set()
    for node_i, node_j in zip(node_ids[:-1], node_ids[1:]):
        data = graph[node_i][node_j]
        weight = float(data["weight"])
        record: dict[str, float | str] = {
            "from": node_i,
            "to": node_j,
            "weight": weight,
            "score": float(data.get("score", weight)),
        }
        if beta is not None:
            record["unnormalized_probability"] = float(np.exp(-beta * weight))
        edge_scores.append(record)
        edges.add(_edge_key(graph, node_i, node_j))

    best_edges = {_edge_key(graph, a, b) for a, b in zip(best_nodes[:-1], best_nodes[1:])}
    edge_jaccard = _jaccard(edges, best_edges)
    node_jaccard = _jaccard(set(node_ids[1:-1]), set(best_nodes[1:-1]))
    total_cost = sum(float(edge["weight"]) for edge in edge_scores)
    absolute_delta = total_cost - best_cost
    relative_delta = (
        absolute_delta / abs(best_cost)
        if best_cost != 0
        else (0.0 if absolute_delta == 0 else None)
    )
    return PathCandidate(
        node_ids=list(node_ids),
        total_cost=float(total_cost),
        absolute_delta=float(absolute_delta),
        relative_delta=None if relative_delta is None else float(relative_delta),
        edge_jaccard=float(edge_jaccard),
        edge_diversity=float(1.0 - edge_jaccard),
        intermediate_node_jaccard=float(node_jaccard),
        global_cost_rank=rank,
        edge_scores=edge_scores,
        metadata={"n_path_nodes": len(node_ids), "n_path_edges": len(edges), "beta": beta},
    )


def _scan_candidate_paths(
    graph: nx.Graph,
    start_id: str,
    end_id: str,
    *,
    beta: float | None = None,
    scan_limit: int = 1000,
    k: int,
    diversity_threshold: float,
    mode: Literal["exact", "diverse", "both"],
) -> tuple[list[PathCandidate], list[str], int]:
    best_nodes = list(nx.shortest_path(graph, start_id, end_id, weight="weight"))
    best_cost = float(nx.shortest_path_length(graph, start_id, end_id, weight="weight"))
    candidates: list[PathCandidate] = []
    seen: set[tuple[str, ...]] = set()
    scan_count = 0
    # The generator is cost ordered, but Dijkstra is deliberately canonical rank 1.
    for raw_path in nx.shortest_simple_paths(graph, start_id, end_id, weight="weight"):
        scan_count += 1
        path = tuple(raw_path)
        if path in seen:
            if scan_count >= scan_limit:
                break
            continue
        seen.add(path)
        candidate = _candidate(graph, list(path), best_nodes, best_cost, scan_count, beta)
        candidates.append(candidate)
        exact_count = len(candidates) + int(tuple(best_nodes) not in seen)
        diverse_count = sum(
            item.node_ids != best_nodes and item.edge_diversity >= diversity_threshold
            for item in candidates
        )
        exact_ready = exact_count >= k
        diverse_ready = diverse_count >= k
        if (
            (mode == "exact" and exact_ready)
            or (mode == "diverse" and diverse_ready)
            or (mode == "both" and exact_ready and diverse_ready)
        ):
            break
        if scan_count >= scan_limit:
            break

    # A shortest_simple_paths tie can differ from shortest_path's tie choice.
    if tuple(best_nodes) not in seen:
        candidates.insert(0, _candidate(graph, best_nodes, best_nodes, best_cost, 1, beta))
    candidates.sort(key=lambda item: (item.total_cost, item.node_ids))
    best_index = next(index for index, item in enumerate(candidates) if item.node_ids == best_nodes)
    candidates.insert(0, candidates.pop(best_index))
    for rank, item in enumerate(candidates, 1):
        item.global_cost_rank = rank
    return candidates, best_nodes, scan_count


def _select_candidate_paths(
    candidates: list[PathCandidate],
    best_nodes: list[str],
    k: int,
    diversity_threshold: float,
    mode: Literal["exact", "diverse"],
) -> list[PathCandidate]:
    if mode == "diverse":
        return [
            item for item in candidates
            if item.node_ids != best_nodes and item.edge_diversity >= diversity_threshold
        ][:k]
    return candidates[:k]


def find_candidate_paths(
    graph: nx.Graph,
    start_id: str,
    end_id: str,
    k: int = 4,
    *,
    beta: float | None = None,
    diversity_threshold: float = 0.20,
    scan_limit: int = 1000,
    mode: Literal["exact", "diverse"] = "exact",
) -> list[PathCandidate]:
    """Return exact or diversity-filtered shortest-simple-path candidates."""
    _validate_search_parameters(k, diversity_threshold, scan_limit)
    if mode not in {"exact", "diverse"}:
        raise ValueError("mode must be 'exact' or 'diverse'")
    _validate_weights(graph)
    candidates, best_nodes, _ = _scan_candidate_paths(
        graph,
        start_id,
        end_id,
        beta=beta,
        scan_limit=scan_limit,
        k=k,
        diversity_threshold=diversity_threshold,
        mode=mode,
    )
    return _select_candidate_paths(candidates, best_nodes, k, diversity_threshold, mode)


def find_top_k_paths(
    graph: nx.Graph,
    start_id: str,
    end_id: str,
    k: int = 4,
    *,
    beta: float | None = None,
    diversity_threshold: float = 0.20,
    scan_limit: int = 1000,
) -> TopKPathsResult:
    """Return exact top-K paths and up to K diverse alternatives."""
    _validate_search_parameters(k, diversity_threshold, scan_limit)
    _validate_weights(graph)
    candidates, best_nodes, scan_count = _scan_candidate_paths(
        graph,
        start_id,
        end_id,
        beta=beta,
        scan_limit=scan_limit,
        k=k,
        diversity_threshold=diversity_threshold,
        mode="both",
    )
    exact = _select_candidate_paths(candidates, best_nodes, k, diversity_threshold, "exact")
    diverse = _select_candidate_paths(candidates, best_nodes, k, diversity_threshold, "diverse")
    return CandidatePathCollection(
        best_path=exact[0],
        exact_paths=exact,
        diverse_paths=diverse,
        settings={
            "k": k,
            "diversity_threshold": diversity_threshold,
            "scan_limit": scan_limit,
            "beta": beta,
            "start_id": start_id,
            "end_id": end_id,
        },
        scan_count=scan_count,
    )
