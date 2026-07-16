"""Run the end-to-end guided conformational path demo.

The demo is intentionally small and CPU-friendly. It either loads two supplied
PDB files or creates toy C-alpha endpoint traces, generates validity-filtered
Gaussian perturbation nodes, builds a weighted graph, finds the Dijkstra
shortest path, and writes inspection artifacts under ``outputs/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guided_conformational_path.generators import DummyPerturbationGenerator
from guided_conformational_path.candidate_report import write_candidate_path_reports
from guided_conformational_path.graph import build_conformation_graph
from guided_conformational_path.io import read_pdb_ca, write_pdb_ca
from guided_conformational_path.nodes import ConformationNode
from guided_conformational_path.path_search import find_shortest_path, find_top_k_paths
from guided_conformational_path.scoring import DistanceHistogramScorer, RMSDScorer
from guided_conformational_path.visualization import plot_graph


def main() -> None:
    """Parse arguments and run the demo pipeline.

    This command is the quickest integration check for the package. It does not
    perform model inference, physical simulation, or trajectory validation.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-a", type=Path, default=None)
    parser.add_argument("--endpoint-b", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--scorer", choices=["rmsd", "distance_histogram"], default="rmsd")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--path-top-k", type=int, default=4)
    parser.add_argument("--min-path-edge-diversity", type=float, default=0.20)
    parser.add_argument("--path-candidate-scan-limit", type=int, default=1000)
    parser.add_argument("--no-candidate-report", action="store_false", dest="write_candidate_report")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Placeholder only; no long-running training or checkpoint resume exists in v1.",
    )
    args = parser.parse_args()

    if args.resume:
        print("Resume requested, but checkpoint/resume is a placeholder in this MVP.")

    endpoint_a, endpoint_b = _load_or_create_endpoints(args.endpoint_a, args.endpoint_b)
    n_from_a = args.n_samples // 2
    n_from_b = args.n_samples - n_from_a

    generator = DummyPerturbationGenerator(seed=args.seed)
    generated_a = generator.generate_from_endpoint(endpoint_a, n_from_a)
    generated_b = generator.generate_from_endpoint(endpoint_b, n_from_b)

    scorer = RMSDScorer() if args.scorer == "rmsd" else DistanceHistogramScorer(metric="l2")
    graph = build_conformation_graph(endpoint_a, endpoint_b, generated_a, generated_b, scorer)
    result = find_shortest_path(graph, endpoint_a.node_id, endpoint_b.node_id, beta=1.0)
    candidates = find_top_k_paths(
        graph,
        endpoint_a.node_id,
        endpoint_b.node_id,
        k=args.path_top_k,
        beta=1.0,
        diversity_threshold=args.min_path_edge_diversity,
        scan_limit=args.path_candidate_scan_limit,
    )

    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    nodes_dir = args.outputs_dir / "nodes"
    nodes_dir.mkdir(parents=True, exist_ok=True)
    for node in [endpoint_a, endpoint_b, *generated_a, *generated_b]:
        write_pdb_ca(nodes_dir / f"{node.node_id}.pdb", node.coords)

    summary_path = args.outputs_dir / "path_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, indent=2)

    plot_graph(graph, result.node_ids, args.outputs_dir / "graph.png")
    if args.write_candidate_report:
        write_candidate_path_reports(args.outputs_dir, graph, candidates)
    print(f"Shortest path: {' -> '.join(result.node_ids)}")
    print(f"Total cost: {result.total_cost:.4f}")
    print(f"Wrote {summary_path} and {args.outputs_dir / 'graph.png'}")


def _load_or_create_endpoints(
    endpoint_a_path: Path | None,
    endpoint_b_path: Path | None,
) -> tuple[ConformationNode, ConformationNode]:
    """Load both endpoint PDBs or create deterministic toy endpoints.

    The CLI requires both endpoint paths or neither. Supplying only one endpoint
    would make the graph path problem ill-defined for this demo, so it raises a
    clear ``ValueError``.
    """
    if endpoint_a_path is not None and endpoint_b_path is not None:
        return (
            ConformationNode("A", "A_endpoint", read_pdb_ca(endpoint_a_path), file_path=endpoint_a_path),
            ConformationNode("B", "B_endpoint", read_pdb_ca(endpoint_b_path), file_path=endpoint_b_path),
        )
    if endpoint_a_path is not None or endpoint_b_path is not None:
        raise ValueError("Provide both --endpoint-a and --endpoint-b, or neither.")

    coords_a, coords_b = _make_toy_endpoints()
    return (
        ConformationNode("A", "A_endpoint", coords_a, metadata={"synthetic": True}),
        ConformationNode("B", "B_endpoint", coords_b, metadata={"synthetic": True}),
    )


def _make_toy_endpoints(n_residues: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Create two smooth synthetic C-alpha traces for quick CPU demos.

    The second endpoint bends the back half of the chain relative to the first.
    This gives the graph a visible endpoint difference without requiring any
    external structure files.
    """
    residue_idx = np.arange(n_residues, dtype=float)
    coords_a = np.column_stack(
        [
            residue_idx * 3.8,
            np.sin(residue_idx * 0.35) * 2.0,
            np.cos(residue_idx * 0.35) * 2.0,
        ]
    )
    hinge = residue_idx >= n_residues // 2
    coords_b = coords_a.copy()
    coords_b[hinge, 1] += np.linspace(0.0, 12.0, int(np.count_nonzero(hinge)))
    coords_b[hinge, 2] += np.linspace(0.0, -6.0, int(np.count_nonzero(hinge)))
    return coords_a, coords_b


if __name__ == "__main__":
    main()
