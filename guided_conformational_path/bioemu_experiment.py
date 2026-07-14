"""Run graph/path analysis on precomputed BioEmu sample bundles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from guided_conformational_path.bioemu_io import load_bioemu_bundle, load_bioemu_manifest
from guided_conformational_path.large_synthetic import (
    LargeSyntheticRun,
    build_and_search,
    load_pdb_endpoints,
    summarize_graph,
)
from guided_conformational_path.validity import check_node_validity


@dataclass(slots=True)
class BioEmuPrecomputedConfig:
    """Configuration for local analysis of a precomputed BioEmu bundle."""

    endpoint_a_path: Path = Path("data/pdb_inputs/A.pdb")
    endpoint_b_path: Path = Path("data/pdb_inputs/B.pdb")
    bioemu_bundle_dir: Path = Path("data/bioemu_outputs/latest")
    max_bioemu_samples: int | None = 200
    k_neighbors: int = 0
    outputs_dir: Path = Path("outputs/bioemu_experiment")
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
    matching_method: str = "exact_then_geometric"
    min_exact_match_fraction: float = 0.8
    geometric_match_distance_cutoff: float = 5.0
    min_geometric_matches: int = 8

    @property
    def networkx_k_neighbors(self) -> int | None:
        """Return the NetworkX graph k-neighbor setting."""
        return None if self.k_neighbors == 0 else self.k_neighbors


def run_bioemu_precomputed_experiment(config: BioEmuPrecomputedConfig) -> LargeSyntheticRun:
    """Load endpoint PDBs plus BioEmu bundle nodes and find an A-to-B path."""
    endpoint_a, endpoint_b = load_pdb_endpoints(
        config.endpoint_a_path,
        config.endpoint_b_path,
        matching_method=config.matching_method,
        min_exact_match_fraction=config.min_exact_match_fraction,
        geometric_match_distance_cutoff=config.geometric_match_distance_cutoff,
        min_geometric_matches=config.min_geometric_matches,
    )
    if endpoint_a.n_residues != endpoint_b.n_residues:
        raise ValueError(
            f"Endpoint residue counts must match; got {endpoint_a.n_residues} and {endpoint_b.n_residues}"
        )

    bioemu_nodes = load_bioemu_bundle(
        config.bioemu_bundle_dir,
        expected_n_residues=endpoint_a.n_residues,
        max_samples=config.max_bioemu_samples,
    )
    graph, result = build_and_search(
        endpoint_a,
        endpoint_b,
        bioemu_nodes,
        config.networkx_k_neighbors,
    )
    manifest = load_bioemu_manifest(config.bioemu_bundle_dir)
    summary = summarize_graph(
        graph,
        result,
        generated_count=len(bioemu_nodes),
        generation_summaries=[
            {
                "phase": "bioemu_precomputed",
                "bundle_dir": str(config.bioemu_bundle_dir),
                "loaded_samples": len(bioemu_nodes),
                "manifest_schema": manifest.get("schema_version"),
                "sequence": manifest.get("sequence", {}),
            }
        ],
        fallback_used=False,
    )
    summary["config"] = _json_ready_config(config)
    summary["input"] = {
        "mode": "bioemu_precomputed",
        "endpoint_a_path": str(config.endpoint_a_path),
        "endpoint_b_path": str(config.endpoint_b_path),
        "bioemu_bundle_dir": str(config.bioemu_bundle_dir),
        "n_residues": endpoint_a.n_residues,
        "matching": endpoint_a.metadata["matching"],
        "bioemu_sequence": manifest.get("sequence", {}),
    }
    summary["endpoint_validity"] = {
        "A": asdict(check_node_validity(endpoint_a)),
        "B": asdict(check_node_validity(endpoint_b)),
    }
    return LargeSyntheticRun(
        endpoint_a=endpoint_a,
        endpoint_b=endpoint_b,
        graph=graph,
        path_result=result,
        summary=summary,
        fallback_used=False,
    )


def _json_ready_config(config: BioEmuPrecomputedConfig) -> dict[str, Any]:
    """Return config values with paths converted to strings."""
    values = asdict(config)
    for key, value in list(values.items()):
        if isinstance(value, Path):
            values[key] = str(value)
    return values
