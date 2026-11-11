"""Graph visualization helpers.

Visualization is intentionally static in the MVP. The goal is to give a quick
inspection artifact for graph topology and selected path membership, not a
publication-quality structural movie.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

_CACHE_DIR = Path(tempfile.gettempdir()) / "guided_conformational_path_cache"
(_CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_DIR / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR / "xdg"))

import matplotlib.pyplot as plt
import networkx as nx


SOURCE_COLORS = {
    "A_endpoint": "#1f77b4",
    "B_endpoint": "#d62728",
    "A_generated": "#8ecae6",
    "B_generated": "#ffb4a2",
    "external": "#bdbdbd",
}


def plot_graph(
    graph: nx.Graph,
    path_node_ids: list[str] | None = None,
    output_path: str | Path = "outputs/graph.png",
    seed: int = 7,
    min_edge_weight: float | None = None,
    max_edge_weight: float | None = None,
    non_path_edge_alpha: float = 0.85,
    path_edge_alpha: float = 0.95,
) -> None:
    """Save a static graph layout PNG with optional shortest-path highlight.

    Nodes are colored by source label and endpoints are drawn larger than
    generated nodes. If ``path_node_ids`` is supplied, edges along that path are
    drawn darker and wider. Non-path edges can be filtered by stored edge
    ``weight`` and/or drawn with low alpha for large dense graphs. Path edges
    are always retained even when they fall outside the filter range.

    In this project edge ``weight`` is a cost, so ``max_edge_weight`` is often
    the useful sparse-view control: it hides high-cost edges while retaining the
    lower-cost neighborhood. ``min_edge_weight`` is included for diagnostics when
    a user explicitly wants to hide very low-cost edges.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path_nodes = set(path_node_ids or [])
    path_edges = set()
    if path_node_ids:
        path_edges = {
            tuple(sorted((node_i, node_j)))
            for node_i, node_j in zip(path_node_ids[:-1], path_node_ids[1:])
        }

    layout_graph = graph.copy()
    for node_i, node_j, attrs in layout_graph.edges(data=True):
        cost = max(float(attrs.get("weight", 1.0)), 1e-9)
        attrs["layout_weight"] = 1.0 / cost
    positions = nx.spring_layout(layout_graph, seed=seed, weight="layout_weight")
    node_colors = [
        SOURCE_COLORS.get(graph.nodes[node_id].get("source", "external"), "#bdbdbd")
        for node_id in graph.nodes
    ]
    node_sizes = [
        420 if graph.nodes[node_id].get("source") in {"A_endpoint", "B_endpoint"} else 180
        for node_id in graph.nodes
    ]
    path_edge_list = []
    non_path_edge_list = []
    for edge in graph.edges:
        edge_key = tuple(sorted(edge))
        if edge_key in path_edges:
            path_edge_list.append(edge)
            continue
        weight = float(graph.edges[edge].get("weight", 0.0))
        if min_edge_weight is not None and weight < min_edge_weight:
            continue
        if max_edge_weight is not None and weight > max_edge_weight:
            continue
        non_path_edge_list.append(edge)

    plt.figure(figsize=(9, 7))
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=non_path_edge_list,
        edge_color="#cccccc",
        width=0.8,
        alpha=non_path_edge_alpha,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=path_edge_list,
        edge_color="#111111",
        width=3.0,
        alpha=path_edge_alpha,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors=["#111111" if node in path_nodes else "#ffffff" for node in graph.nodes],
        linewidths=[1.6 if node in path_nodes else 0.5 for node in graph.nodes],
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def animate_graph_build(*args: Any, **kwargs: Any) -> None:
    """Placeholder for future graph construction animation.

    The placeholder reserves the API location for a future implementation that
    could animate node addition, edge creation, or path refinement. It is not
    implemented because the first version only needs static PNG output.
    """
    raise NotImplementedError("Graph animation is intentionally left as a future extension.")
