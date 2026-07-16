"""Run the local graph pipeline from a dataset pair and Colab BioEmu bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guided_conformational_path.dataset_bioemu_experiment import (
    DatasetBioEmuConfig,
    run_dataset_bioemu_experiment,
)
from guided_conformational_path.large_synthetic import format_run_summary, write_run_outputs


def parse_args() -> argparse.Namespace:
    """Parse dataset/BioEmu adapter and graph-output options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair-dir",
        type=Path,
        required=True,
        help="Exported endpoint-pair directory containing manifest.json.",
    )
    parser.add_argument(
        "--bioemu-bundle-dir",
        type=Path,
        required=True,
        help="Downloaded Colab bundle containing manifest.json and nodes/*.pdb.",
    )
    parser.add_argument("--max-bioemu-samples", type=int, default=200)
    parser.add_argument("--k-neighbors", type=int, default=0)
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs/dataset_bioemu_experiment"),
    )
    parser.add_argument("--write-nodes", action="store_true")
    parser.add_argument("--graph-min-edge-weight", type=float, default=None)
    parser.add_argument("--graph-max-edge-weight", type=float, default=None)
    parser.add_argument("--graph-non-path-edge-alpha", type=float, default=0.125)
    parser.add_argument("--no-pymol", action="store_false", dest="write_pymol")
    parser.add_argument("--no-py3dmol", action="store_false", dest="write_py3dmol")
    parser.add_argument("--no-animation", action="store_false", dest="write_animation")
    parser.add_argument("--animation-frames-per-edge", type=int, default=20)
    parser.add_argument("--animation-fps", type=int, default=10)
    parser.add_argument("--animation-dpi", type=int, default=120)
    parser.set_defaults(write_pymol=True, write_py3dmol=True, write_animation=True)
    return parser.parse_args()


def main() -> None:
    """Run adaptation, graph search, and the standard output writers."""
    args = parse_args()
    config = DatasetBioEmuConfig(
        dataset_pair_dir=args.pair_dir,
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
    )
    run = run_dataset_bioemu_experiment(config)
    write_run_outputs(run, config)
    print(format_run_summary(run, config.outputs_dir))


if __name__ == "__main__":
    main()
