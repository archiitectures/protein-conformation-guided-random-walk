"""Run the large synthetic conformational graph experiment from the CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guided_conformational_path.large_synthetic import (
    LargeSyntheticConfig,
    format_run_summary,
    run_large_synthetic_experiment,
    write_run_outputs,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI options for the large synthetic experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-residues", type=int, default=200)
    parser.add_argument("--n-samples", type=int, default=80)
    parser.add_argument("--noise-scale", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--recursive-max-iterations",
        type=int,
        default=6,
        help="Maximum recursive Gaussian expansion iterations.",
    )
    parser.add_argument(
        "--recursive-top-k",
        type=int,
        default=8,
        help="Number of closest cross-side frontier parents to keep per side.",
    )
    parser.add_argument(
        "--recursive-samples-per-parent",
        type=int,
        default=4,
        help="Gaussian children requested from each selected frontier parent.",
    )
    parser.add_argument("--target-min-intermediates", type=int, default=3)
    parser.add_argument("--target-max-intermediates", type=int, default=4)
    parser.add_argument(
        "--no-fallback",
        action="store_false",
        dest="allow_fallback",
        help="Return the best recursive Gaussian path directly instead of using waypoint fallback.",
    )
    parser.add_argument("--fallback-waypoints", type=int, default=4)
    parser.add_argument("--fallback-samples-per-waypoint", type=int, default=0)
    parser.add_argument("--fallback-noise-scale", type=float, default=0.05)
    parser.add_argument("--k-neighbors", type=int, default=0)
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs/large_synthetic"))
    parser.add_argument("--write-nodes", action="store_true")
    parser.add_argument(
        "--graph-min-edge-weight",
        type=float,
        default=None,
        help="Hide non-path graph edges below this stored edge weight.",
    )
    parser.add_argument(
        "--graph-max-edge-weight",
        type=float,
        default=None,
        help="Hide non-path graph edges above this stored edge weight. Useful because weight is a cost.",
    )
    parser.add_argument(
        "--graph-non-path-edge-alpha",
        type=float,
        default=0.125,
        help="Transparency for non-path graph edges in graph.png.",
    )
    parser.add_argument("--path-top-k", type=int, default=4)
    parser.add_argument("--min-path-edge-diversity", type=float, default=0.20)
    parser.add_argument("--path-candidate-scan-limit", type=int, default=1000)
    parser.add_argument("--num-top-gifs-generated", type=int, default=6)
    parser.add_argument("--num-bottom-gifs-generated", type=int, default=1)
    parser.add_argument("--no-candidate-report", action="store_false", dest="write_candidate_report")
    parser.add_argument("--write-alternative-path-visuals", action="store_true")
    parser.add_argument(
        "--no-animation",
        action="store_false",
        dest="write_animation",
        help="Skip writing the best-path morph GIF.",
    )
    parser.add_argument(
        "--animation-frames-per-edge",
        type=int,
        default=20,
        help="Interpolated frames to generate for each path edge in the morph GIF.",
    )
    parser.add_argument(
        "--animation-fps",
        type=int,
        default=10,
        help="Frames per second for the morph GIF.",
    )
    parser.add_argument(
        "--animation-dpi",
        type=int,
        default=120,
        help="Render DPI for the morph GIF frames.",
    )
    parser.add_argument(
        "--no-pymol",
        action="store_false",
        dest="write_pymol",
        help="Skip writing the PyMOL path-view PDB files and .pml script.",
    )
    parser.add_argument(
        "--no-py3dmol",
        action="store_false",
        dest="write_py3dmol",
        help="Skip writing the standalone py3Dmol/3Dmol.js HTML path view.",
    )
    parser.set_defaults(
        allow_fallback=True,
        write_pymol=True,
        write_py3dmol=True,
        write_animation=True,
        write_candidate_report=True,
        write_alternative_path_visuals=False,
    )
    return parser.parse_args()


def main() -> None:
    """Run the experiment and write output files."""
    args = parse_args()
    config = LargeSyntheticConfig(**vars(args))
    run = run_large_synthetic_experiment(config)
    write_run_outputs(run, config)
    print(format_run_summary(run, config.outputs_dir))


if __name__ == "__main__":
    main()
