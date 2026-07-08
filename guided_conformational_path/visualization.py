"""Graph visualization helpers."""

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
) -> None:
    """Save a static graph layout PNG with optional shortest-path highlight."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path_nodes = set(path_node_ids or [])
    path_edges = set()
    if path_node_ids:
        path_edges = {
            tuple(sorted((node_i, node_j)))
            for node_i, node_j in zip(path_node_ids[:-1], path_node_ids[1:])
        }

    positions = nx.spring_layout(graph, seed=seed, weight="weight")
    node_colors = [
        SOURCE_COLORS.get(graph.nodes[node_id].get("source", "external"), "#bdbdbd")
        for node_id in graph.nodes
    ]
    node_sizes = [
        420 if graph.nodes[node_id].get("source") in {"A_endpoint", "B_endpoint"} else 180
        for node_id in graph.nodes
    ]
    edge_colors = [
        "#111111" if tuple(sorted(edge)) in path_edges else "#cccccc"
        for edge in graph.edges
    ]
    edge_widths = [
        3.0 if tuple(sorted(edge)) in path_edges else 0.8
        for edge in graph.edges
    ]

    plt.figure(figsize=(9, 7))
    nx.draw_networkx_edges(graph, positions, edge_color=edge_colors, width=edge_widths, alpha=0.85)
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
    """Placeholder for future graph construction animation."""
    raise NotImplementedError("Graph animation is intentionally left as a future extension.")
