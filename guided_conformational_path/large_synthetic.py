"""Reusable helpers for the large synthetic conformational graph experiment.

The functions in this module define the synthetic endpoints, route families,
recursive Gaussian sampling phase, fallback logic, graph construction, and output writing.
Scripts and notebooks should import this module instead of duplicating
experiment definitions in their UI layer.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

_CACHE_DIR = Path(tempfile.gettempdir()) / "guided_conformational_path_cache"
(_CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_DIR / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR / "xdg"))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from guided_conformational_path.candidate_report import write_candidate_path_reports
from guided_conformational_path.generators import DummyPerturbationGenerator
from guided_conformational_path.graph import build_conformation_graph
from guided_conformational_path.io import CARecord, read_pdb_ca_records, write_pdb_ca
from guided_conformational_path.nodes import ConformationNode
from guided_conformational_path.path_search import (
    CandidatePathCollection,
    PathResult,
    find_shortest_path,
    find_top_k_paths,
)
from guided_conformational_path.scoring import EdgeScorer, kabsch_rmsd
from guided_conformational_path.validity import check_node_validity
from guided_conformational_path.visualization import plot_graph


@dataclass(slots=True)
class LargeSyntheticConfig:
    """Configuration for the large synthetic graph experiment."""

    n_residues: int = 200
    n_samples: int = 80
    noise_scale: float = 0.15
    seed: int = 7
    recursive_max_iterations: int = 6
    recursive_top_k: int = 8
    recursive_samples_per_parent: int = 4
    target_min_intermediates: int = 3
    target_max_intermediates: int = 4
    allow_fallback: bool = True
    fallback_waypoints: int = 4
    fallback_samples_per_waypoint: int = 0
    fallback_noise_scale: float = 0.05
    k_neighbors: int = 0
    outputs_dir: Path = Path("outputs/large_synthetic")
    write_nodes: bool = False
    write_pymol: bool = True
    write_py3dmol: bool = True
    write_animation: bool = True
    animation_frames_per_edge: int = 20
    animation_fps: int = 10
    animation_dpi: int = 120
    graph_min_edge_weight: float | None = None
    graph_max_edge_weight: float | None = None
    graph_non_path_edge_alpha: float = 0.125
    path_top_k: int = 4
    min_path_edge_diversity: float = 0.20
    path_candidate_scan_limit: int = 1000
    write_candidate_report: bool = True
    write_alternative_path_visuals: bool = False

    @property
    def networkx_k_neighbors(self) -> int | None:
        """Return the NetworkX graph k-neighbor setting."""
        return None if self.k_neighbors == 0 else self.k_neighbors


@dataclass(slots=True)
class LargeSyntheticRun:
    """In-memory result of a large synthetic experiment run."""

    endpoint_a: ConformationNode
    endpoint_b: ConformationNode
    graph: nx.Graph
    path_result: PathResult
    summary: dict[str, Any]
    fallback_used: bool
    candidate_paths: CandidatePathCollection | None = None


@dataclass(slots=True)
class PDBEndpointConfig:
    """Configuration for running the experiment with user-provided PDB endpoints."""

    endpoint_a_path: Path = Path("data/pdb_inputs/A.pdb")
    endpoint_b_path: Path = Path("data/pdb_inputs/B.pdb")
    n_samples: int = 80
    noise_scale: float = 0.15
    seed: int = 7
    recursive_max_iterations: int = 6
    recursive_top_k: int = 8
    recursive_samples_per_parent: int = 4
    target_min_intermediates: int = 3
    target_max_intermediates: int = 4
    allow_fallback: bool = True
    fallback_waypoints: int = 4
    fallback_samples_per_waypoint: int = 0
    fallback_noise_scale: float = 0.05
    k_neighbors: int = 0
    outputs_dir: Path = Path("outputs/pdb_endpoint_experiment")
    write_nodes: bool = False
    write_pymol: bool = True
    write_py3dmol: bool = True
    write_animation: bool = True
    animation_frames_per_edge: int = 20
    animation_fps: int = 10
    animation_dpi: int = 120
    graph_min_edge_weight: float | None = None
    graph_max_edge_weight: float | None = None
    graph_non_path_edge_alpha: float = 0.125
    path_top_k: int = 4
    min_path_edge_diversity: float = 0.20
    path_candidate_scan_limit: int = 1000
    write_candidate_report: bool = True
    write_alternative_path_visuals: bool = False
    matching_method: str = "exact_then_geometric"
    min_exact_match_fraction: float = 0.8
    geometric_match_distance_cutoff: float = 5.0
    min_geometric_matches: int = 8

    @property
    def networkx_k_neighbors(self) -> int | None:
        """Return the NetworkX graph k-neighbor setting."""
        return None if self.k_neighbors == 0 else self.k_neighbors


class SquaredRMSDScorer(EdgeScorer):
    """Transition-style scorer that penalizes one large jump more than substeps."""

    def score(self, node_i: ConformationNode, node_j: ConformationNode) -> float:
        """Return Kabsch RMSD squared."""
        rmsd = kabsch_rmsd(node_i.coords, node_j.coords)
        return float(rmsd * rmsd)


def make_synthetic_endpoints(n_residues: int = 200) -> tuple[ConformationNode, ConformationNode]:
    """Create compact globular synthetic A/B C-alpha endpoints."""
    if n_residues < 8:
        raise ValueError("n_residues must be at least 8 for this synthetic experiment")

    coords_a = make_globular_ca_trace(n_residues, seed=11)
    coords_b = make_globular_ca_trace(n_residues, seed=23)

    endpoint_a = ConformationNode("A", "A_endpoint", coords_a, metadata={"synthetic": True})
    endpoint_b = ConformationNode("B", "B_endpoint", coords_b, metadata={"synthetic": True})
    return endpoint_a, endpoint_b


def make_globular_ca_trace(n_residues: int, seed: int = 11) -> np.ndarray:
    """Create a compact 3D C-alpha trace with approximately 3.8 A spacing."""
    rng = np.random.default_rng(seed)
    coords = np.zeros((n_residues, 3), dtype=float)
    direction = _unit_vector(rng.normal(size=3))
    target_radius = max(18.0, 4.2 * n_residues ** (1.0 / 3.0))
    non_neighbor_clearance = 2.4

    for idx in range(1, n_residues):
        best_direction = direction
        best_coord = coords[idx - 1] + 3.8 * direction
        best_score = float("inf")
        for _ in range(100):
            center_pull = -coords[idx - 1] / max(target_radius, np.linalg.norm(coords[idx - 1]) + 1e-8)
            candidate_direction = _unit_vector(
                0.55 * direction
                + 0.65 * rng.normal(size=3)
                + 0.45 * center_pull
            )
            candidate_coord = coords[idx - 1] + 3.8 * candidate_direction
            if idx <= 2:
                min_non_neighbor_distance = float("inf")
            else:
                min_non_neighbor_distance = float(
                    np.min(np.linalg.norm(coords[: idx - 1] - candidate_coord, axis=1))
                )
            radius_penalty = max(0.0, np.linalg.norm(candidate_coord) - target_radius * 1.6)
            clash_penalty = max(0.0, non_neighbor_clearance - min_non_neighbor_distance)
            score = 1_000.0 * clash_penalty + radius_penalty
            if score < best_score:
                best_score = score
                best_direction = candidate_direction
                best_coord = candidate_coord
            if min_non_neighbor_distance > non_neighbor_clearance and radius_penalty == 0.0:
                break
        direction = best_direction
        coords[idx] = best_coord

    coords -= coords.mean(axis=0)
    return coords


def _smooth_trace(coords: np.ndarray, n_passes: int) -> np.ndarray:
    """Smooth a C-alpha trace while preserving endpoints approximately."""
    smoothed = np.asarray(coords, dtype=float).copy()
    for _ in range(n_passes):
        middle = 0.25 * smoothed[:-2] + 0.5 * smoothed[1:-1] + 0.25 * smoothed[2:]
        smoothed[1:-1] = middle
    return _renormalize_bond_lengths(smoothed, bond_length=3.8)


def _renormalize_bond_lengths(coords: np.ndarray, bond_length: float) -> np.ndarray:
    """Rebuild a trace from directions so consecutive distances are stable."""
    rebuilt = np.zeros_like(coords, dtype=float)
    rebuilt[0] = coords[0]
    for idx in range(1, coords.shape[0]):
        step = coords[idx] - coords[idx - 1]
        rebuilt[idx] = rebuilt[idx - 1] + bond_length * _unit_vector(step)
    rebuilt -= rebuilt.mean(axis=0)
    return rebuilt


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    """Return a unit vector, using x-axis fallback for near-zero vectors."""
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0])
    return np.asarray(vector, dtype=float) / norm


def load_pdb_endpoints(
    endpoint_a_path: str | Path,
    endpoint_b_path: str | Path,
    matching_method: str = "exact_then_geometric",
    min_exact_match_fraction: float = 0.8,
    geometric_match_distance_cutoff: float = 5.0,
    min_geometric_matches: int = 8,
) -> tuple[ConformationNode, ConformationNode]:
    """Load endpoint C-alpha coordinates from PDB files with residue matching."""
    path_a = Path(endpoint_a_path)
    path_b = Path(endpoint_b_path)
    records_a = read_pdb_ca_records(path_a)
    records_b = read_pdb_ca_records(path_b)
    coords_a, coords_b, metadata = match_pdb_ca_records(
        records_a,
        records_b,
        matching_method=matching_method,
        min_exact_match_fraction=min_exact_match_fraction,
        geometric_match_distance_cutoff=geometric_match_distance_cutoff,
        min_geometric_matches=min_geometric_matches,
    )
    endpoint_a = ConformationNode(
        "A",
        "A_endpoint",
        coords_a,
        metadata={"pdb_input": True, "matching": metadata},
        file_path=path_a,
    )
    endpoint_b = ConformationNode(
        "B",
        "B_endpoint",
        coords_b,
        metadata={"pdb_input": True, "matching": metadata},
        file_path=path_b,
    )
    return endpoint_a, endpoint_b


def match_pdb_ca_records(
    records_a: list[CARecord],
    records_b: list[CARecord],
    matching_method: str = "exact_then_geometric",
    min_exact_match_fraction: float = 0.8,
    geometric_match_distance_cutoff: float = 5.0,
    min_geometric_matches: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Match C-alpha records by residue IDs, then optionally by geometry."""
    if matching_method not in {"exact", "geometric", "exact_then_geometric"}:
        raise ValueError("matching_method must be 'exact', 'geometric', or 'exact_then_geometric'")
    exact_a, exact_b, exact_keys = _exact_record_match(records_a, records_b)
    exact_fraction = len(exact_keys) / max(1, min(len(records_a), len(records_b)))

    if matching_method in {"exact", "exact_then_geometric"} and exact_fraction >= min_exact_match_fraction:
        metadata = {
            "method_used": "exact",
            "original_count_a": len(records_a),
            "original_count_b": len(records_b),
            "matched_count": len(exact_keys),
            "exact_overlap_count": len(exact_keys),
            "exact_match_fraction": exact_fraction,
        }
        return _coords_from_records(exact_a), _coords_from_records(exact_b), metadata

    if matching_method == "exact":
        raise ValueError(
            "Exact residue matching found insufficient overlap: "
            f"{len(exact_keys)} shared residues, fraction {exact_fraction:.3f}"
        )

    geometric_a, geometric_b, distances = _geometric_record_match(
        records_a,
        records_b,
        geometric_match_distance_cutoff,
    )
    if len(geometric_a) < min_geometric_matches:
        raise ValueError(
            "Geometric residue matching found too few mutual C-alpha matches: "
            f"{len(geometric_a)} < {min_geometric_matches}"
        )
    metadata = {
        "method_used": "geometric",
        "original_count_a": len(records_a),
        "original_count_b": len(records_b),
        "matched_count": len(geometric_a),
        "exact_overlap_count": len(exact_keys),
        "exact_match_fraction": exact_fraction,
        "geometric_cutoff": geometric_match_distance_cutoff,
        "geometric_mean_distance": float(np.mean(distances)) if distances else None,
        "geometric_max_distance": float(np.max(distances)) if distances else None,
    }
    return _coords_from_records(geometric_a), _coords_from_records(geometric_b), metadata


def _exact_record_match(
    records_a: list[CARecord],
    records_b: list[CARecord],
) -> tuple[list[CARecord], list[CARecord], list[tuple[str, int, str]]]:
    """Match records by chain/residue/insertion-code identity."""
    by_key_a = {record.residue_key: record for record in records_a}
    by_key_b = {record.residue_key: record for record in records_b}
    shared_keys = sorted(set(by_key_a) & set(by_key_b), key=lambda key: (key[0], key[1], key[2]))
    return (
        [by_key_a[key] for key in shared_keys],
        [by_key_b[key] for key in shared_keys],
        shared_keys,
    )


def _geometric_record_match(
    records_a: list[CARecord],
    records_b: list[CARecord],
    cutoff: float,
) -> tuple[list[CARecord], list[CARecord], list[float]]:
    """Match C-alpha records by mutual nearest neighbors after Kabsch alignment."""
    coords_a = _coords_from_records(records_a)
    coords_b = _coords_from_records(records_b)
    n_init = min(coords_a.shape[0], coords_b.shape[0])
    aligned_a, aligned_b = _kabsch_aligned_pair(coords_a[:n_init], coords_b[:n_init])
    full_a = (coords_a - coords_a[:n_init].mean(axis=0)) @ _kabsch_rotation(coords_a[:n_init], coords_b[:n_init])
    full_b = coords_b - coords_b[:n_init].mean(axis=0)
    if aligned_a.shape[0] != n_init or aligned_b.shape[0] != n_init:
        raise ValueError("Internal alignment error during geometric matching")

    tree_b = cKDTree(full_b)
    dist_ab, idx_ab = tree_b.query(full_a, k=1)
    tree_a = cKDTree(full_a)
    _, idx_ba = tree_a.query(full_b, k=1)

    pairs: list[tuple[int, int, float]] = []
    for idx_a, (idx_b, distance) in enumerate(zip(idx_ab, dist_ab)):
        if idx_ba[int(idx_b)] == idx_a and float(distance) <= cutoff:
            pairs.append((idx_a, int(idx_b), float(distance)))
    pairs.sort(key=lambda pair: pair[0])
    return (
        [records_a[idx_a] for idx_a, _, _ in pairs],
        [records_b[idx_b] for _, idx_b, _ in pairs],
        [distance for _, _, distance in pairs],
    )


def _kabsch_aligned_pair(coords_a: np.ndarray, coords_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return centered/aligned A and centered B for geometric matching."""
    rotation = _kabsch_rotation(coords_a, coords_b)
    centered_a = coords_a - coords_a.mean(axis=0)
    centered_b = coords_b - coords_b.mean(axis=0)
    return centered_a @ rotation, centered_b


def _kabsch_rotation(coords_a: np.ndarray, coords_b: np.ndarray) -> np.ndarray:
    """Return Kabsch rotation from A to B."""
    centered_a = coords_a - coords_a.mean(axis=0)
    centered_b = coords_b - coords_b.mean(axis=0)
    covariance = centered_a.T @ centered_b
    u_mat, _, vt_mat = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(u_mat @ vt_mat))
    return u_mat @ correction @ vt_mat


def _coords_from_records(records: list[CARecord]) -> np.ndarray:
    """Return coordinate array from parsed C-alpha records."""
    return np.asarray([record.coord for record in records], dtype=float)


def make_waypoint_family(
    endpoint_a: ConformationNode,
    endpoint_b: ConformationNode,
    n_waypoints: int,
    family: str,
    detour_scale: float,
) -> list[ConformationNode]:
    """Create ordered intermediate waypoints between endpoints."""
    if n_waypoints <= 0:
        raise ValueError("n_waypoints must be positive")

    n_residues = endpoint_a.n_residues
    t = np.linspace(0.0, 1.0, n_residues)
    detour = np.column_stack(
        [
            np.zeros(n_residues),
            detour_scale * np.sin(np.pi * t),
            detour_scale * 0.6 * np.sin(2.0 * np.pi * t),
        ]
    )

    nodes: list[ConformationNode] = []
    for idx in range(1, n_waypoints + 1):
        fraction = idx / (n_waypoints + 1)
        coords = (1.0 - fraction) * endpoint_a.coords + fraction * endpoint_b.coords
        coords = coords + np.sin(np.pi * fraction) * detour
        nodes.append(
            ConformationNode(
                node_id=f"{family}_{idx:02d}",
                source="external",
                coords=coords,
                metadata={"family": family, "waypoint_index": idx, "fraction": fraction},
            )
        )
    return nodes


def generate_gaussian_nodes(
    endpoints: list[ConformationNode],
    n_samples_total: int,
    noise_scale: float,
    seed: int,
) -> tuple[list[ConformationNode], list[dict[str, Any]]]:
    """Generate Gaussian nodes spread as evenly as possible across endpoints."""
    if n_samples_total <= 0:
        return [], []

    generator = DummyPerturbationGenerator(
        noise_scale=noise_scale,
        seed=seed,
        max_attempts_per_sample=20,
    )
    generated: list[ConformationNode] = []
    summaries: list[dict[str, Any]] = []
    base_count = n_samples_total // len(endpoints)
    remainder = n_samples_total % len(endpoints)

    for idx, endpoint in enumerate(endpoints):
        n_for_endpoint = base_count + (1 if idx < remainder else 0)
        nodes = generator.generate_from_endpoint(endpoint, n_for_endpoint)
        generated.extend(nodes)
        summaries.append({"endpoint": endpoint.node_id, **generator.last_generation_summary})
    return generated, summaries


def generate_recursive_gaussian_nodes(
    endpoint_a: ConformationNode,
    endpoint_b: ConformationNode,
    n_samples_total: int,
    noise_scale: float,
    seed: int,
    max_iterations: int = 6,
    top_k: int = 8,
    samples_per_parent: int = 4,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[ConformationNode], list[dict[str, Any]]]:
    """Generate nodes by recursively perturbing selected A-side/B-side frontiers.

    The sampler starts from endpoint A and endpoint B, generates Gaussian
    children from each side, then repeatedly selects frontier parents from the
    closest cross-side A/B pairs. The total number of accepted generated nodes is
    capped by ``n_samples_total`` so graph construction remains bounded.

    If ``progress_callback`` is provided, it is called once after each attempted
    recursive iteration with a JSON-serializable status dictionary. This lets
    notebooks display tqdm progress without making tqdm a runtime dependency.
    """
    if n_samples_total <= 0:
        return [], []
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if samples_per_parent <= 0:
        raise ValueError("samples_per_parent must be positive")

    generator = DummyPerturbationGenerator(
        noise_scale=noise_scale,
        seed=seed,
        max_attempts_per_sample=20,
    )
    scorer = SquaredRMSDScorer()
    generated_by_side: dict[str, list[ConformationNode]] = {"A": [], "B": []}
    frontier_by_side: dict[str, list[ConformationNode]] = {"A": [endpoint_a], "B": [endpoint_b]}
    counters = {"A": 0, "B": 0}
    summaries: list[dict[str, Any]] = []

    for iteration in range(1, max_iterations + 1):
        if sum(len(nodes) for nodes in generated_by_side.values()) >= n_samples_total:
            break

        tasks = _interleaved_frontier_tasks(frontier_by_side)
        accepted_this_iteration = 0
        for side, parent in tasks:
            remaining = n_samples_total - sum(len(nodes) for nodes in generated_by_side.values())
            if remaining <= 0:
                break
            requested = min(samples_per_parent, remaining)
            source = "A_generated" if side == "A" else "B_generated"
            children = generator.generate_from_endpoint(parent, requested, source=source)
            renamed_children: list[ConformationNode] = []
            for child in children:
                node_id = f"{side}_rec_{counters[side]:03d}"
                counters[side] += 1
                metadata = {
                    **child.metadata,
                    "recursive_iteration": iteration,
                    "recursive_side": side,
                    "parent": parent.node_id,
                    "recursive_parent_source": parent.source,
                }
                renamed_children.append(
                    ConformationNode(
                        node_id=node_id,
                        source=source,
                        coords=child.coords,
                        metadata=metadata,
                        file_path=child.file_path,
                    )
                )

            generated_by_side[side].extend(renamed_children)
            accepted_this_iteration += len(renamed_children)
            summaries.append(
                {
                    "phase": "recursive_gaussian",
                    "iteration": iteration,
                    "side": side,
                    "parent": parent.node_id,
                    **generator.last_generation_summary,
                    "returned": len(renamed_children),
                }
            )

        frontier_by_side = select_recursive_frontiers(
            generated_by_side["A"],
            generated_by_side["B"],
            endpoint_a,
            endpoint_b,
            scorer,
            top_k,
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "recursive_gaussian_iteration",
                    "iteration": iteration,
                    "max_iterations": max_iterations,
                    "accepted_this_iteration": accepted_this_iteration,
                    "returned_total": sum(len(nodes) for nodes in generated_by_side.values()),
                    "requested_total": n_samples_total,
                    "frontier_a": len(frontier_by_side["A"]),
                    "frontier_b": len(frontier_by_side["B"]),
                }
            )
        if accepted_this_iteration == 0:
            break

    all_generated = [*generated_by_side["A"], *generated_by_side["B"]]
    summaries.append(
        {
            "phase": "recursive_gaussian_summary",
            "requested_total": n_samples_total,
            "returned_total": len(all_generated),
            "max_iterations": max_iterations,
            "top_k": top_k,
            "samples_per_parent": samples_per_parent,
            "returned_by_side": {
                "A": len(generated_by_side["A"]),
                "B": len(generated_by_side["B"]),
            },
        }
    )
    return all_generated, summaries


def select_recursive_frontiers(
    a_nodes: list[ConformationNode],
    b_nodes: list[ConformationNode],
    endpoint_a: ConformationNode,
    endpoint_b: ConformationNode,
    scorer: EdgeScorer,
    top_k: int,
) -> dict[str, list[ConformationNode]]:
    """Select A/B frontier parents from the closest cross-side generated pairs."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not a_nodes or not b_nodes:
        return {
            "A": a_nodes[:top_k] if a_nodes else [endpoint_a],
            "B": b_nodes[:top_k] if b_nodes else [endpoint_b],
        }

    scored_pairs: list[tuple[float, ConformationNode, ConformationNode]] = []
    for node_a in a_nodes:
        for node_b in b_nodes:
            scored_pairs.append((scorer.score(node_a, node_b), node_a, node_b))
    scored_pairs.sort(key=lambda item: item[0])

    selected_a: dict[str, ConformationNode] = {}
    selected_b: dict[str, ConformationNode] = {}
    for _, node_a, node_b in scored_pairs:
        if len(selected_a) < top_k:
            selected_a.setdefault(node_a.node_id, node_a)
        if len(selected_b) < top_k:
            selected_b.setdefault(node_b.node_id, node_b)
        if len(selected_a) >= top_k and len(selected_b) >= top_k:
            break

    return {"A": list(selected_a.values()), "B": list(selected_b.values())}


def _interleaved_frontier_tasks(
    frontier_by_side: dict[str, list[ConformationNode]],
) -> list[tuple[str, ConformationNode]]:
    """Return side/parent tasks interleaved A, B, A, B for budget fairness."""
    tasks: list[tuple[str, ConformationNode]] = []
    max_frontier = max(len(frontier_by_side["A"]), len(frontier_by_side["B"]))
    for idx in range(max_frontier):
        for side in ("A", "B"):
            if idx < len(frontier_by_side[side]):
                tasks.append((side, frontier_by_side[side][idx]))
    return tasks


def build_and_search(
    endpoint_a: ConformationNode,
    endpoint_b: ConformationNode,
    generated_nodes: list[ConformationNode],
    k_neighbors: int | None,
) -> tuple[nx.Graph, PathResult]:
    """Build a fully scored graph with squared-RMSD edges and find A-to-B path."""
    graph = build_conformation_graph(
        endpoint_a=endpoint_a,
        endpoint_b=endpoint_b,
        generated_a=generated_nodes,
        generated_b=[],
        scorer=SquaredRMSDScorer(),
        lambda_validity=0.1,
        k_neighbors=k_neighbors,
    )
    result = find_shortest_path(graph, endpoint_a.node_id, endpoint_b.node_id, beta=1.0)
    return graph, result


def build_route_constrained_graph(
    endpoint_a: ConformationNode,
    endpoint_b: ConformationNode,
    nodes: list[ConformationNode],
) -> tuple[nx.Graph, PathResult]:
    """Build fallback graph as separate route-family chains."""
    scorer = SquaredRMSDScorer()
    graph = nx.Graph()
    all_nodes = [endpoint_a, endpoint_b, *nodes]
    validity_by_id = {node.node_id: check_node_validity(node) for node in all_nodes}

    for node in all_nodes:
        validity = validity_by_id[node.node_id]
        graph.add_node(
            node.node_id,
            conformation=node,
            source=node.source,
            validity_score=validity.score,
            valid=validity.valid,
            validity_messages=validity.messages,
        )

    families = sorted(
        {
            node.metadata["family"]
            for node in nodes
            if "family" in node.metadata and "waypoint_index" in node.metadata
        }
    )
    for family in families:
        family_nodes = sorted(
            [
                node
                for node in nodes
                if node.metadata.get("family") == family and "waypoint_index" in node.metadata
            ],
            key=lambda node: int(node.metadata["waypoint_index"]),
        )
        route_nodes = [endpoint_a, *family_nodes, endpoint_b]
        for node_i, node_j in zip(route_nodes[:-1], route_nodes[1:]):
            raw_score = scorer.score(node_i, node_j)
            validity_penalty = validity_by_id[node_i.node_id].score + validity_by_id[node_j.node_id].score
            weight = raw_score + 0.1 * validity_penalty
            graph.add_edge(node_i.node_id, node_j.node_id, weight=weight, score=raw_score)

    result = find_shortest_path(graph, endpoint_a.node_id, endpoint_b.node_id, beta=1.0)
    return graph, result


def path_intermediate_count(result: PathResult) -> int:
    """Return number of non-endpoint nodes in a path result."""
    return max(0, len(result.node_ids) - 2)


def route_family_costs(graph: nx.Graph) -> dict[str, float]:
    """Return total chain cost for each route family in a constrained graph."""
    families = sorted(
        {
            attrs["conformation"].metadata["family"]
            for _, attrs in graph.nodes(data=True)
            if "family" in attrs["conformation"].metadata
            and "waypoint_index" in attrs["conformation"].metadata
        }
    )
    costs: dict[str, float] = {}
    for family in families:
        family_ids = [
            node_id
            for node_id, attrs in graph.nodes(data=True)
            if attrs["conformation"].metadata.get("family") == family
            and "waypoint_index" in attrs["conformation"].metadata
        ]
        family_ids.sort(
            key=lambda node_id: int(graph.nodes[node_id]["conformation"].metadata["waypoint_index"])
        )
        route_ids = ["A", *family_ids, "B"]
        cost = 0.0
        complete = True
        for node_i, node_j in zip(route_ids[:-1], route_ids[1:]):
            if not graph.has_edge(node_i, node_j):
                complete = False
                break
            cost += float(graph[node_i][node_j]["weight"])
        if complete:
            costs[family] = cost
    return costs


def should_use_fallback(
    result: PathResult,
    generated_count: int,
    target_min_intermediates: int,
    target_max_intermediates: int,
) -> bool:
    """Decide whether generated output met the intended path shape."""
    n_intermediates = path_intermediate_count(result)
    if generated_count == 0:
        return True
    return not (target_min_intermediates <= n_intermediates <= target_max_intermediates)


def make_fallback_nodes(
    endpoint_a: ConformationNode,
    endpoint_b: ConformationNode,
    n_waypoints: int,
    samples_per_waypoint: int,
    noise_scale: float,
    seed: int,
) -> tuple[list[ConformationNode], list[dict[str, Any]]]:
    """Create waypoint families and optional Gaussian neighborhoods around them."""
    waypoint_families = [
        ("low_energy", 0.0),
        ("medium_detour", 8.0),
        ("high_detour", 16.0),
    ]
    waypoints: list[ConformationNode] = []
    for family, detour_scale in waypoint_families:
        waypoints.extend(make_waypoint_family(endpoint_a, endpoint_b, n_waypoints, family, detour_scale))

    noised_nodes, summaries = generate_gaussian_nodes(
        waypoints,
        n_samples_total=len(waypoints) * samples_per_waypoint,
        noise_scale=noise_scale,
        seed=seed,
    )
    return [*waypoints, *noised_nodes], summaries


def summarize_graph(
    graph: nx.Graph,
    result: PathResult,
    generated_count: int,
    generation_summaries: list[dict[str, Any]],
    fallback_used: bool,
) -> dict[str, Any]:
    """Create a JSON-serializable experiment summary."""
    valid_count = sum(1 for _, attrs in graph.nodes(data=True) if attrs.get("valid"))
    path_families = [
        graph.nodes[node_id]["conformation"].metadata.get("family", graph.nodes[node_id].get("source"))
        for node_id in result.node_ids
    ]
    return {
        "fallback_used": fallback_used,
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "n_generated_nodes": generated_count,
        "n_valid_nodes": valid_count,
        "path_node_ids": result.node_ids,
        "path_families": path_families,
        "path_intermediates": path_intermediate_count(result),
        "total_cost": result.total_cost,
        "route_family_costs": route_family_costs(graph),
        "generation_summaries": generation_summaries,
    }


def run_endpoint_experiment(
    endpoint_a: ConformationNode,
    endpoint_b: ConformationNode,
    config: LargeSyntheticConfig | PDBEndpointConfig,
    input_metadata: dict[str, Any] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> LargeSyntheticRun:
    """Run recursive Gaussian generation first, then fallback if needed."""
    recursive_nodes, recursive_summaries = generate_recursive_gaussian_nodes(
        endpoint_a,
        endpoint_b,
        n_samples_total=config.n_samples,
        noise_scale=config.noise_scale,
        seed=config.seed,
        max_iterations=config.recursive_max_iterations,
        top_k=config.recursive_top_k,
        samples_per_parent=config.recursive_samples_per_parent,
        progress_callback=progress_callback,
    )
    recursive_graph, recursive_result = build_and_search(
        endpoint_a,
        endpoint_b,
        recursive_nodes,
        config.networkx_k_neighbors,
    )
    recursive_path_missed_target = should_use_fallback(
        recursive_result,
        len(recursive_nodes),
        config.target_min_intermediates,
        config.target_max_intermediates,
    )
    use_fallback = config.allow_fallback and recursive_path_missed_target

    selected_graph = recursive_graph
    selected_result = recursive_result
    selected_nodes = recursive_nodes
    selected_summaries = recursive_summaries

    if use_fallback:
        fallback_nodes, fallback_summaries = make_fallback_nodes(
            endpoint_a,
            endpoint_b,
            n_waypoints=config.fallback_waypoints,
            samples_per_waypoint=config.fallback_samples_per_waypoint,
            noise_scale=config.fallback_noise_scale,
            seed=config.seed + 1,
        )
        selected_graph, selected_result = build_route_constrained_graph(
            endpoint_a,
            endpoint_b,
            fallback_nodes,
        )
        selected_nodes = fallback_nodes
        selected_summaries = fallback_summaries

    summary = summarize_graph(
        selected_graph,
        selected_result,
        generated_count=len(selected_nodes),
        generation_summaries=selected_summaries,
        fallback_used=use_fallback,
    )
    summary["config"] = {
        **asdict(config),
        "outputs_dir": str(config.outputs_dir),
    }
    if isinstance(config, PDBEndpointConfig):
        summary["config"]["endpoint_a_path"] = str(config.endpoint_a_path)
        summary["config"]["endpoint_b_path"] = str(config.endpoint_b_path)
    if input_metadata:
        summary["input"] = input_metadata
    summary["recursive_path_missed_target"] = recursive_path_missed_target
    summary["fallback_allowed"] = config.allow_fallback
    summary["recursive_gaussian"] = summarize_graph(
        recursive_graph,
        recursive_result,
        generated_count=len(recursive_nodes),
        generation_summaries=recursive_summaries,
        fallback_used=False,
    )
    summary["endpoint_validity"] = {
        "A": asdict(check_node_validity(endpoint_a)),
        "B": asdict(check_node_validity(endpoint_b)),
    }
    candidate_paths = find_top_k_paths(
        selected_graph,
        endpoint_a.node_id,
        endpoint_b.node_id,
        k=config.path_top_k,
        beta=1.0,
        diversity_threshold=config.min_path_edge_diversity,
        scan_limit=config.path_candidate_scan_limit,
    )

    return LargeSyntheticRun(
        endpoint_a=endpoint_a,
        endpoint_b=endpoint_b,
        graph=selected_graph,
        path_result=selected_result,
        summary=summary,
        fallback_used=use_fallback,
        candidate_paths=candidate_paths,
    )


def run_large_synthetic_experiment(
    config: LargeSyntheticConfig,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> LargeSyntheticRun:
    """Run the experiment with generated synthetic endpoints."""
    endpoint_a, endpoint_b = make_synthetic_endpoints(config.n_residues)
    return run_endpoint_experiment(
        endpoint_a,
        endpoint_b,
        config,
        input_metadata={"mode": "synthetic", "n_residues": config.n_residues},
        progress_callback=progress_callback,
    )


def run_pdb_endpoint_experiment(
    config: PDBEndpointConfig,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> LargeSyntheticRun:
    """Run the experiment with endpoint conformations loaded from PDB files."""
    endpoint_a, endpoint_b = load_pdb_endpoints(
        config.endpoint_a_path,
        config.endpoint_b_path,
        matching_method=config.matching_method,
        min_exact_match_fraction=config.min_exact_match_fraction,
        geometric_match_distance_cutoff=config.geometric_match_distance_cutoff,
        min_geometric_matches=config.min_geometric_matches,
    )
    return run_endpoint_experiment(
        endpoint_a,
        endpoint_b,
        config,
        input_metadata={
            "mode": "pdb",
            "endpoint_a_path": str(config.endpoint_a_path),
            "endpoint_b_path": str(config.endpoint_b_path),
            "n_residues": endpoint_a.n_residues,
            "matching": endpoint_a.metadata["matching"],
        },
        progress_callback=progress_callback,
    )


def write_outputs(
    output_dir: Path,
    graph: nx.Graph,
    result: PathResult,
    summary: dict[str, Any],
    write_nodes: bool,
    graph_min_edge_weight: float | None = None,
    graph_max_edge_weight: float | None = None,
    graph_non_path_edge_alpha: float = 0.125,
) -> None:
    """Write graph, path, summary, and optional PDB-like nodes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with (output_dir / "path_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, indent=2)
    plot_graph(
        graph,
        result.node_ids,
        output_dir / "graph.png",
        min_edge_weight=graph_min_edge_weight,
        max_edge_weight=graph_max_edge_weight,
        non_path_edge_alpha=graph_non_path_edge_alpha,
    )

    if write_nodes:
        nodes_dir = output_dir / "nodes"
        nodes_dir.mkdir(parents=True, exist_ok=True)
        for node_id, attrs in graph.nodes(data=True):
            node = attrs["conformation"]
            write_pdb_ca(nodes_dir / f"{node_id}.pdb", node.coords)


def write_pymol_path_view(output_dir: Path, run: LargeSyntheticRun) -> Path:
    """Write A/B/path-node PDBs and a PyMOL script for visual inspection.

    The repository does not depend on PyMOL. This function emits plain PDB files
    and a ``.pml`` script that can be opened in a local PyMOL installation.
    """
    pymol_dir = output_dir / "pymol_path"
    pymol_dir.mkdir(parents=True, exist_ok=True)
    node_paths: list[tuple[str, Path]] = []

    for order_idx, node_id in enumerate(run.path_result.node_ids):
        node = run.graph.nodes[node_id]["conformation"]
        safe_id = node_id.replace("/", "_").replace(" ", "_")
        path = pymol_dir / f"{order_idx:02d}_{safe_id}.pdb"
        write_pdb_ca(path, node.coords)
        node_paths.append((safe_id, path))

    pml_path = pymol_dir / "view_path.pml"
    colors = [
        "marine",
        "cyan",
        "tv_green",
        "yellow",
        "orange",
        "salmon",
        "red",
        "magenta",
        "purpleblue",
        "gray70",
    ]
    with pml_path.open("w", encoding="utf-8") as handle:
        handle.write("reinitialize\n")
        handle.write("bg_color white\n")
        for idx, (safe_id, path) in enumerate(node_paths):
            object_name = f"path_{idx:02d}_{safe_id}"
            color = colors[min(idx, len(colors) - 1)]
            handle.write(f"load {path.name}, {object_name}\n")
            handle.write(f"hide everything, {object_name}\n")
            handle.write(f"show spheres, {object_name}\n")
            handle.write(f"set sphere_scale, 0.45, {object_name}\n")
            handle.write(f"color {color}, {object_name}\n")
        handle.write("set ray_opaque_background, off\n")
        handle.write("set depth_cue, 0\n")
        handle.write("orient\n")
        handle.write("zoom\n")
        handle.write("png path_nodes.png, width=1600, height=1000, ray=1\n")
    return pml_path


def path_node_pdb_blocks(run: LargeSyntheticRun) -> list[tuple[str, str]]:
    """Return ``(node_id, pdb_text)`` blocks for nodes on the selected path."""
    blocks: list[tuple[str, str]] = []
    for node_id in run.path_result.node_ids:
        node = run.graph.nodes[node_id]["conformation"]
        lines = []
        for idx, (x_coord, y_coord, z_coord) in enumerate(node.coords, start=1):
            lines.append(
                f"ATOM  {idx:5d}  CA  ALA A{idx:4d}    "
                f"{x_coord:8.3f}{y_coord:8.3f}{z_coord:8.3f}"
                f"  1.00  0.00           C"
            )
        lines.append("END")
        blocks.append((node_id, "\n".join(lines)))
    return blocks


def path_node_centers(run: LargeSyntheticRun) -> list[tuple[str, tuple[float, float, float]]]:
    """Return centroid positions for conformations on the selected graph path.

    Each graph node is a full conformation, not a single atom. The centroid is a
    compact representative position, but synthetic conformations are centered,
    so centroids can overlap. Use ``path_node_marker_positions`` when a visibly
    separated graph-path overlay is needed.
    """
    centers: list[tuple[str, tuple[float, float, float]]] = []
    for node_id in run.path_result.node_ids:
        node = run.graph.nodes[node_id]["conformation"]
        center = np.mean(node.coords, axis=0)
        centers.append((node_id, (float(center[0]), float(center[1]), float(center[2]))))
    return centers


def path_node_marker_positions(run: LargeSyntheticRun) -> list[tuple[str, tuple[float, float, float]]]:
    """Return visible marker positions for graph nodes in the 3D path view.

    The marker positions form a simple rail above the structural overlay. This
    intentionally separates graph semantics from atom coordinates: the colored
    traces are the conformations, while the rail shows selected graph nodes and
    best-path edges in an easy-to-read location.
    """
    path_ids = run.path_result.node_ids
    path_coords = [
        run.graph.nodes[node_id]["conformation"].coords
        for node_id in path_ids
    ]
    stacked = np.vstack(path_coords)
    min_corner = np.min(stacked, axis=0)
    max_corner = np.max(stacked, axis=0)
    span = np.maximum(max_corner - min_corner, 1.0)
    y_coord = float(max_corner[1] + 0.18 * span[1] + 8.0)
    z_coord = float(max_corner[2] + 0.18 * span[2] + 8.0)
    if len(path_ids) == 1:
        x_positions = [float(np.mean(stacked[:, 0]))]
    else:
        x_padding = 0.08 * span[0] + 4.0
        x_positions = np.linspace(
            float(min_corner[0] - x_padding),
            float(max_corner[0] + x_padding),
            len(path_ids),
        )
    return [
        (node_id, (float(x_coord), y_coord, z_coord))
        for node_id, x_coord in zip(path_ids, x_positions)
    ]


def path_node_coordinate_payloads(
    run: LargeSyntheticRun,
    separate_conformations: bool = True,
    spacing_scale: float = 1.35,
    min_clearance: float = 12.0,
) -> list[dict[str, Any]]:
    """Return JSON-ready coordinates for full path conformations.

    The standalone HTML uses this payload to draw all C-alpha residues, backbone
    connections within each conformation, and sparse residue motion traces
    between consecutive conformations. By default, coordinates are translated
    for display so selected path structures are visible separately; this does
    not alter graph/scoring coordinates stored in ``run``.

    Paths of four nodes or fewer are shown as a linear row. Longer paths are
    arranged around a circle, with A and B shown as a close endpoint pair above
    the ring. Display translations enforce a minimum center-to-center distance
    based on the largest conformation radius, so rendered proteins do not
    overlap in the 3D view.
    """
    path_nodes = [run.graph.nodes[node_id]["conformation"] for node_id in run.path_result.node_ids]
    path_coords = [node.coords for node in path_nodes]
    layout_mode = "circular" if len(path_coords) > 4 else "linear"
    conformation_radii = [_conformation_display_radius(coords) for coords in path_coords]
    max_conformation_radius = max(conformation_radii, default=0.0)
    min_center_distance = max(2.0 * max_conformation_radius + min_clearance, min_clearance)
    if path_coords:
        stacked = np.vstack(path_coords)
        span = np.maximum(np.max(stacked, axis=0) - np.min(stacked, axis=0), 1.0)
        step = max(float(span[0] * spacing_scale + min_clearance), min_center_distance)
        origin_shift = -0.5 * step * (len(path_coords) - 1)
        n_intermediates = max(1, len(path_coords) - 2)
        if len(path_coords) > 2:
            # Adjacent intermediates on the ring should be at least one display
            # diameter plus clearance apart. The endpoint pair occupies the
            # top sector, so intermediates are distributed over the remaining
            # sectors using n + 1 angular slots.
            angle_step = 2.0 * np.pi / max(2, n_intermediates + 1)
            radius_from_chord = min_center_distance / max(2.0 * np.sin(angle_step / 2.0), 1e-6)
        else:
            radius_from_chord = min_center_distance
        circle_radius = max(
            step * 0.72,
            float(max(span[0], span[1]) * 1.15 + min_clearance),
            radius_from_chord,
        )
        endpoint_pair_offset = 0.5 * min_center_distance
    else:
        step = 0.0
        origin_shift = 0.0
        circle_radius = 0.0
        endpoint_pair_offset = 0.0

    payloads: list[dict[str, Any]] = []
    for idx, (node_id, coords) in enumerate(zip(run.path_result.node_ids, path_coords)):
        if not separate_conformations:
            translation = np.zeros(3, dtype=float)
        elif layout_mode == "circular":
            if idx == 0:
                translation = np.array(
                    [-endpoint_pair_offset, circle_radius + min_center_distance, 0.0],
                    dtype=float,
                )
            elif idx == len(path_coords) - 1:
                translation = np.array(
                    [endpoint_pair_offset, circle_radius + min_center_distance, 0.0],
                    dtype=float,
                )
            else:
                n_intermediates = max(1, len(path_coords) - 2)
                angle = (np.pi / 2.0) - (2.0 * np.pi * idx / (n_intermediates + 1))
                translation = np.array(
                    [circle_radius * np.cos(angle), circle_radius * np.sin(angle), 0.0],
                    dtype=float,
                )
        else:
            translation = np.array(
                [origin_shift + idx * step, 0.0, 0.0],
                dtype=float,
            )
        display_coords = coords + translation
        payloads.append(
            {
                "node_id": node_id,
                "layout_mode": layout_mode,
                "display_radius": float(conformation_radii[idx]),
                "minimum_center_distance": float(min_center_distance),
                "display_translation": {
                    "x": float(translation[0]),
                    "y": float(translation[1]),
                    "z": float(translation[2]),
                },
                "coords": [
                    {"x": float(x_coord), "y": float(y_coord), "z": float(z_coord)}
                    for x_coord, y_coord, z_coord in display_coords
                ],
            }
        )
    return payloads


def _conformation_display_radius(coords: np.ndarray) -> float:
    """Return maximum residue distance from a conformation centroid."""
    if coords.size == 0:
        return 0.0
    centered = coords - np.mean(coords, axis=0)
    return float(np.max(np.linalg.norm(centered, axis=1)))


def write_py3dmol_path_html(output_dir: Path, run: LargeSyntheticRun) -> Path:
    """Write a standalone 3Dmol.js HTML view of A, B, and path nodes.

    The notebook helper uses the Python ``py3Dmol`` package directly. This HTML
    output uses the same 3Dmol.js viewer technology but can be opened in a
    browser without running a Python notebook.
    """
    html_path = output_dir / "py3dmol_path_view.html"
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = [
        "0x1f77b4",
        "0x17becf",
        "0x2ca02c",
        "0xffdf00",
        "0xff7f0e",
        "0xd62728",
        "0x9467bd",
        "0x8c564b",
        "0x7f7f7f",
    ]
    models = [
        {
            "node_id": node_id,
            "color": colors[min(idx, len(colors) - 1)],
        }
        for idx, node_id in enumerate(run.path_result.node_ids)
    ]
    coordinate_payloads = path_node_coordinate_payloads(run)
    sparse_trace_stride = max(1, run.endpoint_a.n_residues // 10)
    medium_trace_stride = max(1, run.endpoint_a.n_residues // 20)
    model_json = json.dumps(models)
    coordinate_json = json.dumps(coordinate_payloads)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>py3Dmol Path View</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; }}
    #viewer-region {{ position: relative; z-index: 1; clear: both; margin-top: 12px; }}
    #viewer {{ position: relative; z-index: 1; width: 100%; height: 720px; border: 1px solid #ddd; overflow: hidden; }}
    #viewer canvas {{ max-width: 100%; }}
    #controls {{ position: relative; z-index: 20; display: flex; flex-wrap: wrap; gap: 14px; align-items: end; margin: 16px 0; padding: 12px; background: #f8f8f8; border: 1px solid #ddd; }}
    label {{ display: flex; flex-direction: column; gap: 4px; font-size: 14px; }}
    select {{ min-width: 180px; padding: 5px 7px; }}
    #status {{ position: relative; z-index: 20; margin: 12px 0; padding: 10px 12px; background: #f5f5f5; border-left: 4px solid #999; }}
    code {{ background: #f5f5f5; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Selected A-to-B Path Structure View</h1>
  <p>This view overlays every C-alpha residue for A, B, and each selected path node from the NetworkX path.</p>
  <p>Short paths are translated side-by-side. Paths longer than four nodes are arranged around a circle, with A and B displayed as a close endpoint pair. Use the controls to switch between atom spheres, backbone sticks plus nodes, or both. Motion traces are intentionally sparse so the view does not become a wall of inter-residue edges.</p>
  <p>Path: <code>{' -> '.join(run.path_result.node_ids)}</code></p>
  <div id="controls">
    <label>
      Representation
      <select id="representation">
        <option value="sticks_nodes" selected>Sticks + nodes</option>
        <option value="spheres">Atom spheres only</option>
        <option value="both">Both</option>
      </select>
    </label>
    <label>
      Motion traces
      <select id="trace-density">
        <option value="sparse" selected>Sparse</option>
        <option value="medium">Medium</option>
        <option value="none">None</option>
      </select>
    </label>
  </div>
  <div id="status">Loading 3Dmol.js...</div>
  <div id="viewer-region">
    <div id="viewer"></div>
  </div>
  <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
  <script>
    const models = {model_json};
    const conformations = {coordinate_json};
    const layoutMode = conformations[0]?.layout_mode ?? "linear";
    const traceStrides = {{
      sparse: {sparse_trace_stride},
      medium: {medium_trace_stride}
    }};
    const status = document.getElementById("status");
    const representationControl = document.getElementById("representation");
    const traceDensityControl = document.getElementById("trace-density");
    let viewer = null;

    function setStatus(message, isError = false) {{
      status.textContent = message;
      status.style.borderLeftColor = isError ? "#b00020" : "#267326";
      status.style.background = isError ? "#fff0f0" : "#f1fff1";
    }}

    function traceStride() {{
      const density = traceDensityControl.value;
      return density === "none" ? null : traceStrides[density];
    }}

    function renderPathView() {{
      try {{
        if (!window.$3Dmol) {{
          setStatus("3Dmol.js did not load. Check internet access or CDN blocking for https://3Dmol.org/build/3Dmol-min.js.", true);
          return;
        }}
        const representation = representationControl.value;
        const stride = traceStride();
        document.getElementById("viewer").innerHTML = "";
        viewer = $3Dmol.createViewer("viewer", {{ backgroundColor: "white" }});
        conformations.forEach((entry, confIdx) => {{
          const color = models[Math.min(confIdx, models.length - 1)].color;
          if (representation === "spheres" || representation === "both") {{
            entry.coords.forEach((coord) => {{
              viewer.addSphere({{
                center: coord,
                radius: 0.24,
                color: color,
                opacity: representation === "both" ? 0.38 : 0.7
              }});
            }});
          }}
          if (representation === "sticks_nodes" || representation === "both") {{
            entry.coords.forEach((coord, residueIdx) => {{
              if (residueIdx % 5 === 0 || residueIdx === 0 || residueIdx === entry.coords.length - 1) {{
                viewer.addSphere({{
                  center: coord,
                  radius: 0.32,
                  color: color,
                  opacity: 0.72
                }});
              }}
            }});
            for (let residueIdx = 0; residueIdx < entry.coords.length - 1; residueIdx += 1) {{
              viewer.addCylinder({{
                start: entry.coords[residueIdx],
                end: entry.coords[residueIdx + 1],
                radius: confIdx === 0 || confIdx === conformations.length - 1 ? 0.18 : 0.13,
                color: color,
                fromCap: 1,
                toCap: 1,
                opacity: confIdx === 0 || confIdx === conformations.length - 1 ? 0.82 : 0.48
              }});
            }}
          }}
          const labelAnchor = entry.coords[0];
          viewer.addLabel(entry.node_id, {{
            position: labelAnchor,
            fontSize: 13,
            fontColor: "black",
            backgroundColor: "white",
            backgroundOpacity: 0.8,
            inFront: true
          }});
        }});
        let traceCount = 0;
        if (stride !== null) {{
          for (let confIdx = 0; confIdx < conformations.length - 1; confIdx += 1) {{
            const current = conformations[confIdx].coords;
            const next = conformations[confIdx + 1].coords;
            const nResidues = Math.min(current.length, next.length);
            for (let residueIdx = 0; residueIdx < nResidues; residueIdx += stride) {{
              viewer.addCylinder({{
                start: current[residueIdx],
                end: next[residueIdx],
                radius: 0.05,
                color: "0x222222",
                fromCap: 1,
                toCap: 1,
                opacity: 0.22
              }});
              traceCount += 1;
            }}
          }}
        }}
        viewer.zoomTo();
        viewer.render();
        const traceText = stride === null ? "no motion traces" : `${{traceCount}} motion traces, every ${{stride}} residues`;
        setStatus(`Rendered ${{models.length}} separated full conformations using ${{layoutMode}} layout, ${{conformations[0]?.coords.length ?? 0}} residues each, ${{traceText}}.`);
      }} catch (error) {{
        console.error(error);
        setStatus(`Render failed: ${{error.message}}`, true);
      }}
    }}

    window.addEventListener("load", renderPathView);
    representationControl.addEventListener("change", renderPathView);
    traceDensityControl.addEventListener("change", renderPathView);
  </script>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    return html_path


def path_morph_frames(
    run: LargeSyntheticRun,
    frames_per_edge: int = 20,
    *,
    min_duration_weight: float = 0.75,
    max_duration_weight: float = 3.0,
) -> list[dict[str, Any]]:
    """Return interpolated C-alpha backbone frames along the selected path.

    Each frame linearly interpolates between two consecutive best-path nodes.
    The returned frames preserve the path-node order from ``run.path_result``.
    """
    if frames_per_edge <= 0:
        raise ValueError("frames_per_edge must be positive")
    if min_duration_weight <= 0.0:
        raise ValueError("min_duration_weight must be positive")
    if max_duration_weight < min_duration_weight:
        raise ValueError("max_duration_weight must be greater than or equal to min_duration_weight")

    node_ids = list(run.path_result.node_ids)
    if len(node_ids) < 2:
        raise ValueError("path morph animation requires at least two path nodes")

    path_nodes = [run.graph.nodes[node_id]["conformation"] for node_id in node_ids]
    edge_movements = [
        float(np.sqrt(np.mean(np.sum((node_i.coords - node_j.coords) ** 2, axis=1))))
        for node_i, node_j in zip(path_nodes[:-1], path_nodes[1:])
    ]
    positive_movements = [movement for movement in edge_movements if movement > 0.0]
    reference_movement = float(np.mean(positive_movements)) if positive_movements else 1.0
    path_frames: list[dict[str, Any]] = []
    for edge_idx, (node_i, node_j) in enumerate(zip(path_nodes[:-1], path_nodes[1:])):
        edge_movement = edge_movements[edge_idx]
        raw_duration_weight = edge_movement / reference_movement if reference_movement > 0.0 else 1.0
        duration_weight = float(np.clip(raw_duration_weight, min_duration_weight, max_duration_weight))
        fractions = np.linspace(0.0, 1.0, frames_per_edge, endpoint=True)
        for frame_idx, fraction in enumerate(fractions):
            coords = (1.0 - fraction) * node_i.coords + fraction * node_j.coords
            path_frames.append(
                {
                    "edge_index": edge_idx,
                    "segment_count": len(path_nodes) - 1,
                    "frame_index": frame_idx,
                    "global_frame_index": edge_idx * frames_per_edge + frame_idx,
                    "frames_per_edge": frames_per_edge,
                    "fraction": float(fraction),
                    "edge_movement": edge_movement,
                    "duration_weight": duration_weight,
                    "start_node_id": node_i.node_id,
                    "end_node_id": node_j.node_id,
                    "title": (
                        f"Segment {edge_idx + 1}/{len(path_nodes) - 1}: "
                        f"{node_i.node_id} -> {node_j.node_id} "
                        f"({frame_idx + 1}/{frames_per_edge})"
                    ),
                    "coords": coords,
                }
            )
    total_frames = len(path_frames)
    playback_denominator = float(
        np.sum([frame["duration_weight"] for frame in path_frames[:-1]], dtype=float)
    )
    cumulative_playback_weight = 0.0
    for frame_idx, frame in enumerate(path_frames):
        progress = (
            float(cumulative_playback_weight / playback_denominator)
            if total_frames > 1 and playback_denominator > 0.0
            else 1.0
        )
        frame["global_frame_index"] = frame_idx
        frame["global_frame_count"] = total_frames
        frame["progress"] = progress
        frame["color"] = _path_morph_progress_color(progress)
        cumulative_playback_weight += float(frame["duration_weight"])
    return path_frames


def _path_morph_progress_color(progress: float) -> str:
    """Return the blue-to-red path color for a normalized animation progress."""
    clipped = float(np.clip(progress, 0.0, 1.0))
    blue = np.array([31, 119, 180], dtype=float)
    red = np.array([214, 39, 40], dtype=float)
    rgb = np.rint((1.0 - clipped) * blue + clipped * red).astype(int)
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _path_morph_axis_limits(run: LargeSyntheticRun) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Return fixed axis limits that cover the full selected path."""
    path_nodes = [run.graph.nodes[node_id]["conformation"] for node_id in run.path_result.node_ids]
    stacked = np.vstack([node.coords for node in path_nodes])
    min_corner = np.min(stacked, axis=0)
    max_corner = np.max(stacked, axis=0)
    center = 0.5 * (min_corner + max_corner)
    span = np.maximum(max_corner - min_corner, 1.0)
    half_range = 0.5 * float(np.max(span)) + 4.0
    return (
        (float(center[0] - half_range), float(center[0] + half_range)),
        (float(center[1] - half_range), float(center[1] + half_range)),
        (float(center[2] - half_range), float(center[2] + half_range)),
    )


def _render_path_morph_frame(
    ax: Any,
    coords: np.ndarray,
    axis_limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    title: str,
    color: str,
) -> None:
    """Render a single 3D backbone frame with fixed axes."""
    ax.clear()
    ax.plot(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        color=color,
        linewidth=2.2,
    )
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        color=color,
        s=10,
        alpha=0.9,
    )
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_axis_off()
    ax.text2D(
        0.5,
        -0.02,
        title,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color="#111827",
    )


def write_path_morph_gif(
    output_dir: Path,
    run: LargeSyntheticRun,
    frames_per_edge: int = 20,
    fps: int = 10,
    dpi: int = 120,
    *,
    min_duration_weight: float = 0.75,
    max_duration_weight: float = 3.0,
) -> Path:
    """Write a GIF that morphs the selected best-path C-alpha backbone."""
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    frames = path_morph_frames(
        run,
        frames_per_edge=frames_per_edge,
        min_duration_weight=min_duration_weight,
        max_duration_weight=max_duration_weight,
    )
    axis_limits = _path_morph_axis_limits(run)
    fig = plt.figure(figsize=(6.2, 6.2))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.15)
    ax = fig.add_subplot(111, projection="3d")
    gif_frames: list[Image.Image] = []
    frame_durations_ms: list[int] = []

    try:
        for frame in frames:
            _render_path_morph_frame(
                ax,
                frame["coords"],
                axis_limits,
                frame["title"],
                frame["color"],
            )
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=dpi, facecolor="white")
            buffer.seek(0)
            frame_image = Image.open(buffer).convert("P", palette=Image.ADAPTIVE)
            gif_frames.append(frame_image)
            base_duration_ms = max(1, int(round(1000 / max(fps, 1))))
            frame_durations_ms.append(
                max(1, int(round(base_duration_ms * float(frame["duration_weight"]))))
            )
            buffer.close()
        if not gif_frames:
            raise RuntimeError("No animation frames were generated")
        gif_path = output_dir / "path_morph.gif"
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=frame_durations_ms,
            loop=0,
            optimize=False,
        )
        return gif_path
    finally:
        plt.close(fig)


def display_py3dmol_path(run: LargeSyntheticRun, width: int = 900, height: int = 650) -> Any:
    """Return a notebook py3Dmol viewer for full path conformations."""
    try:
        import py3Dmol
    except ImportError as exc:
        raise ImportError(
            "py3Dmol is not installed. Run `UV_CACHE_DIR=.uv-cache uv sync --extra notebook`."
        ) from exc

    colors = [
        "blue",
        "cyan",
        "green",
        "yellow",
        "orange",
        "red",
        "purple",
        "magenta",
        "gray",
    ]
    view = py3Dmol.view(width=width, height=height)
    coordinate_payloads = path_node_coordinate_payloads(run)
    residue_stride = max(1, run.endpoint_a.n_residues // 20)
    for conf_idx, payload in enumerate(coordinate_payloads):
        color = colors[min(conf_idx, len(colors) - 1)]
        coords = payload["coords"]
        for coord in coords:
            view.addSphere(
                {
                    "center": coord,
                    "radius": 0.22,
                    "color": color,
                    "opacity": 0.55,
                }
            )
        for start, end in zip(coords[:-1], coords[1:]):
            view.addCylinder(
                {
                    "start": start,
                    "end": end,
                    "radius": 0.18 if conf_idx in {0, len(coordinate_payloads) - 1} else 0.13,
                    "color": color,
                    "fromCap": 1,
                    "toCap": 1,
                    "opacity": 0.82 if conf_idx in {0, len(coordinate_payloads) - 1} else 0.48,
                }
            )
        view.addLabel(
            str(payload["node_id"]),
            {
                "position": coords[0],
                "fontSize": 13,
                "fontColor": "black",
                "backgroundColor": "white",
                "backgroundOpacity": 0.75,
                "inFront": True,
            },
        )
    for current, next_payload in zip(coordinate_payloads[:-1], coordinate_payloads[1:]):
        current_coords = current["coords"]
        next_coords = next_payload["coords"]
        n_residues = min(len(current_coords), len(next_coords))
        for residue_idx in range(0, n_residues, residue_stride):
            view.addCylinder(
                {
                    "start": current_coords[residue_idx],
                    "end": next_coords[residue_idx],
                    "radius": 0.055,
                    "color": "black",
                    "fromCap": 1,
                    "toCap": 1,
                    "opacity": 0.28,
                }
            )
    view.addLabel(
        f"Residue motion traces every {residue_stride} residues",
        {
            "position": coordinate_payloads[0]["coords"][0],
            "fontSize": 10,
            "fontColor": "white",
            "backgroundColor": "black",
            "backgroundOpacity": 0.55,
            "inFront": True,
        },
        )
    view.setBackgroundColor("white")
    view.zoomTo()
    return view


def write_run_outputs(run: LargeSyntheticRun, config: LargeSyntheticConfig) -> None:
    """Write output files for a completed large synthetic run."""
    write_outputs(
        output_dir=config.outputs_dir,
        graph=run.graph,
        result=run.path_result,
        summary=run.summary,
        write_nodes=config.write_nodes,
        graph_min_edge_weight=config.graph_min_edge_weight,
        graph_max_edge_weight=config.graph_max_edge_weight,
        graph_non_path_edge_alpha=config.graph_non_path_edge_alpha,
    )
    if config.write_animation:
        gif_path = write_path_morph_gif(
            config.outputs_dir,
            run,
            frames_per_edge=config.animation_frames_per_edge,
            fps=config.animation_fps,
            dpi=config.animation_dpi,
        )
        run.summary["path_morph_gif"] = str(gif_path)
    if config.write_pymol:
        pml_path = write_pymol_path_view(config.outputs_dir, run)
        run.summary["pymol_path_view"] = str(pml_path)
    if config.write_py3dmol:
        html_path = write_py3dmol_path_html(config.outputs_dir, run)
        run.summary["py3dmol_path_view"] = str(html_path)
    if run.candidate_paths is not None and config.write_candidate_report:
        candidate_json, candidate_html = write_candidate_path_reports(
            config.outputs_dir,
            run.graph,
            run.candidate_paths,
        )
        run.summary["candidate_paths"] = {
            "schema_version": "candidate_paths_v1",
            "json": candidate_json.name,
            "html": candidate_html.name,
            "exact_count": len(run.candidate_paths.exact_paths),
            "diverse_count": len(run.candidate_paths.diverse_paths),
        }
    if run.candidate_paths is not None and config.write_alternative_path_visuals:
        alternative_paths = _write_alternative_path_visuals(run, config)
        run.summary["alternative_path_visuals"] = alternative_paths
    with (config.outputs_dir / "experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(run.summary, handle, indent=2)


def _write_alternative_path_visuals(
    run: LargeSyntheticRun,
    config: LargeSyntheticConfig,
) -> list[dict[str, Any]]:
    """Write requested heavy visuals once for each unique non-best path."""
    if run.candidate_paths is None:
        return []
    best_nodes = tuple(run.path_result.node_ids)
    unique_candidates = {}
    for candidate in [*run.candidate_paths.exact_paths, *run.candidate_paths.diverse_paths]:
        node_key = tuple(candidate.node_ids)
        if node_key != best_nodes:
            unique_candidates.setdefault(node_key, candidate)

    written: list[dict[str, Any]] = []
    for candidate in sorted(unique_candidates.values(), key=lambda item: item.global_cost_rank):
        destination = config.outputs_dir / "alternative_paths" / f"candidate_{candidate.global_cost_rank:03d}"
        alternative_run = LargeSyntheticRun(
            endpoint_a=run.endpoint_a,
            endpoint_b=run.endpoint_b,
            graph=run.graph,
            path_result=PathResult(
                node_ids=list(candidate.node_ids),
                total_cost=candidate.total_cost,
                edge_scores=[dict(edge) for edge in candidate.edge_scores],
                metadata=dict(candidate.metadata),
            ),
            summary={},
            fallback_used=run.fallback_used,
            candidate_paths=None,
        )
        artifacts: dict[str, Any] = {
            "global_cost_rank": candidate.global_cost_rank,
            "node_ids": candidate.node_ids,
            "output_dir": str(destination),
        }
        if config.write_animation:
            artifacts["path_morph_gif"] = str(
                write_path_morph_gif(
                    destination,
                    alternative_run,
                    frames_per_edge=config.animation_frames_per_edge,
                    fps=config.animation_fps,
                    dpi=config.animation_dpi,
                )
            )
        if config.write_pymol:
            artifacts["pymol_path_view"] = str(write_pymol_path_view(destination, alternative_run))
        if config.write_py3dmol:
            artifacts["py3dmol_path_view"] = str(
                write_py3dmol_path_html(destination, alternative_run)
            )
        written.append(artifacts)
    return written


def format_run_summary(run: LargeSyntheticRun, outputs_dir: Path) -> str:
    """Format a concise terminal/notebook summary for a run."""
    lines = [
        f"Residues: {run.endpoint_a.n_residues}",
        f"Fallback used: {run.fallback_used}",
        f"Graph: {run.graph.number_of_nodes()} nodes, {run.graph.number_of_edges()} edges",
        f"Path intermediates: {path_intermediate_count(run.path_result)}",
        f"Shortest path: {' -> '.join(run.path_result.node_ids)}",
        f"Total cost: {run.path_result.total_cost:.4f}",
    ]
    route_costs = route_family_costs(run.graph)
    if route_costs:
        lines.append("Route family costs:")
        lines.extend(f"  {family}: {cost:.4f}" for family, cost in route_costs.items())
    lines.append(f"Wrote {outputs_dir / 'experiment_summary.json'} and {outputs_dir / 'graph.png'}")
    path_morph_gif = run.summary.get("path_morph_gif")
    if path_morph_gif:
        lines.append(f"Wrote path morph GIF: {path_morph_gif}")
    pymol_path = run.summary.get("pymol_path_view")
    if pymol_path:
        lines.append(f"Wrote PyMOL path view script: {pymol_path}")
    py3dmol_path = run.summary.get("py3dmol_path_view")
    if py3dmol_path:
        lines.append(f"Wrote py3Dmol path view HTML: {py3dmol_path}")
    candidate_paths = run.summary.get("candidate_paths")
    if candidate_paths:
        lines.append(
            "Wrote candidate path reports: "
            f"{outputs_dir / candidate_paths['json']} and {outputs_dir / candidate_paths['html']}"
        )
    return "\n".join(lines)
