"""Run local graph analysis on a precomputed BioEmu output bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guided_conformational_path.bioemu_experiment import (
    BioEmuPrecomputedConfig,
    run_bioemu_precomputed_experiment,
)
from guided_conformational_path.large_synthetic import format_run_summary, write_run_outputs


def parse_args() -> argparse.Namespace:
    """Parse CLI options for precomputed BioEmu graph analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-a", type=Path, default=Path("data/pdb_inputs/A.pdb"))
    parser.add_argument("--endpoint-b", type=Path, default=Path("data/pdb_inputs/B.pdb"))
    parser.add_argument(
        "--bioemu-bundle-dir",
        type=Path,
        default=Path("data/bioemu_outputs/latest"),
        help="Directory containing BioEmu manifest.json and normalized PDB sample files.",
    )
    parser.add_argument("--max-bioemu-samples", type=int, default=200)
    parser.add_argument("--k-neighbors", type=int, default=0)
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs/bioemu_experiment"))
    parser.add_argument("--write-nodes", action="store_true")
    parser.add_argument("--graph-min-edge-weight", type=float, default=None)
    parser.add_argument("--graph-max-edge-weight", type=float, default=None)
    parser.add_argument("--graph-non-path-edge-alpha", type=float, default=0.125)
    parser.add_argument("--matching-method", default="exact_then_geometric")
    parser.add_argument("--min-exact-match-fraction", type=float, default=0.8)
    parser.add_argument("--geometric-match-distance-cutoff", type=float, default=5.0)
    parser.add_argument("--min-geometric-matches", type=int, default=8)
    parser.add_argument("--no-pymol", action="store_false", dest="write_pymol")
    parser.add_argument("--no-py3dmol", action="store_false", dest="write_py3dmol")
    parser.add_argument("--no-animation", action="store_false", dest="write_animation")
    parser.add_argument("--animation-frames-per-edge", type=int, default=20)
    parser.add_argument("--animation-fps", type=int, default=10)
    parser.add_argument("--animation-dpi", type=int, default=120)
    parser.set_defaults(write_pymol=True, write_py3dmol=True, write_animation=True)
    return parser.parse_args()


def main() -> None:
    """Run BioEmu precomputed analysis and write output files."""
    args = parse_args()
    config = BioEmuPrecomputedConfig(
        endpoint_a_path=args.endpoint_a,
        endpoint_b_path=args.endpoint_b,
        bioemu_bundle_dir=args.bioemu_bundle_dir,
        max_bioemu_samples=args.max_bioemu_samples,
        k_neighbors=args.k_neighbors,
        outputs_dir=args.outputs_dir,
        write_nodes=args.write_nodes,
        write_pymol=args.write_pymol,
        write_py3dmol=args.write_py3dmol,
        write_animation=args.write_animation,
        animation_frames_per_edge=args.animation_frames_per_edge,
        animation_fps=args.animation_fps,
        animation_dpi=args.animation_dpi,
        graph_min_edge_weight=args.graph_min_edge_weight,
        graph_max_edge_weight=args.graph_max_edge_weight,
        graph_non_path_edge_alpha=args.graph_non_path_edge_alpha,
        matching_method=args.matching_method,
        min_exact_match_fraction=args.min_exact_match_fraction,
        geometric_match_distance_cutoff=args.geometric_match_distance_cutoff,
        min_geometric_matches=args.min_geometric_matches,
    )
    run = run_bioemu_precomputed_experiment(config)
    write_run_outputs(run, config)
    print(format_run_summary(run, config.outputs_dir))


if __name__ == "__main__":
    main()
