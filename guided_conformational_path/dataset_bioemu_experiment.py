"""Run graph analysis from an exported endpoint pair and Colab BioEmu bundle."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from guided_conformational_path.dataset_bioemu_adapter import (
    load_adapted_bioemu_nodes,
    load_dataset_pair_selection,
)
from guided_conformational_path.large_synthetic import (
    LargeSyntheticRun,
    build_and_search,
    summarize_graph,
)
from guided_conformational_path.validity import check_node_validity


@dataclass(slots=True)
class DatasetBioEmuConfig:
    """Configuration for the dataset-pair to BioEmu graph handoff."""

    dataset_pair_dir: Path
    bioemu_bundle_dir: Path
    max_bioemu_samples: int | None = 200
    k_neighbors: int = 0
    outputs_dir: Path = Path("outputs/dataset_bioemu_experiment")
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

    @property
    def networkx_k_neighbors(self) -> int | None:
        """Translate zero from the CLI into a fully connected graph."""
        return None if self.k_neighbors == 0 else self.k_neighbors


def run_dataset_bioemu_experiment(config: DatasetBioEmuConfig) -> LargeSyntheticRun:
    """Adapt a Colab bundle to a dataset pair and find its shortest graph path."""
    selection = load_dataset_pair_selection(config.dataset_pair_dir)
    bioemu_nodes = load_adapted_bioemu_nodes(
        selection,
        config.bioemu_bundle_dir,
        max_samples=config.max_bioemu_samples,
    )
    graph, result = build_and_search(
        selection.endpoint_a,
        selection.endpoint_b,
        bioemu_nodes,
        config.networkx_k_neighbors,
    )
    generation_summary = {
        "phase": "dataset_bioemu_adapted",
        "bundle_dir": str(config.bioemu_bundle_dir),
        "loaded_samples": len(bioemu_nodes),
        "selected_sequence_length": len(selection.selected_sequence),
        "selected_ca_count": len(selection.residue_positions),
        "mapping_source": selection.mapping_source,
        "sequence_verified": True,
    }
    summary = summarize_graph(
        graph,
        result,
        generated_count=len(bioemu_nodes),
        generation_summaries=[generation_summary],
        fallback_used=False,
    )
    summary["config"] = _json_ready_config(config)
    summary["input"] = {
        "mode": "dataset_bioemu",
        "dataset_pair_dir": str(selection.pair_dir),
        "dataset_pair_manifest": str(selection.manifest_path),
        "bioemu_bundle_dir": str(config.bioemu_bundle_dir),
        "protein_id": selection.protein_id,
        "pair_id": selection.pair_id,
        "selected_sequence_length": len(selection.selected_sequence),
        "common_ca_count": len(selection.residue_positions),
        "residue_positions": selection.residue_positions.tolist(),
        "mapping_source": selection.mapping_source,
        "sequence_verified": True,
    }
    summary["endpoint_validity"] = {
        "A": asdict(check_node_validity(selection.endpoint_a)),
        "B": asdict(check_node_validity(selection.endpoint_b)),
    }
    return LargeSyntheticRun(
        endpoint_a=selection.endpoint_a,
        endpoint_b=selection.endpoint_b,
        graph=graph,
        path_result=result,
        summary=summary,
        fallback_used=False,
    )


def _json_ready_config(config: DatasetBioEmuConfig) -> dict[str, Any]:
    """Return dataclass values with paths converted for JSON output."""
    values = asdict(config)
    for key, value in list(values.items()):
        if isinstance(value, Path):
            values[key] = str(value)
    return values
