from __future__ import annotations

from guided_conformational_path.bioemu_runner import run_bioemu_from_fasta as public_run_bioemu_from_fasta
from guided_conformational_path.dataset_bioemu_adapter import (
    load_adapted_bioemu_nodes as public_load_adapted_bioemu_nodes,
    load_dataset_pair_selection as public_load_dataset_pair_selection,
)
from guided_conformational_path.dataset_bioemu_experiment import (
    DatasetBioEmuConfig as PublicDatasetBioEmuConfig,
    run_dataset_bioemu_experiment as public_run_dataset_bioemu_experiment,
)
from guided_conformational_path.large_synthetic import write_run_outputs as public_write_run_outputs

from parallel_bioemu_tools import collector, worker


def test_cross_project_public_imports_are_wired() -> None:
    assert worker.run_bioemu_from_fasta is public_run_bioemu_from_fasta
    assert collector.load_dataset_pair_selection is public_load_dataset_pair_selection
    assert collector.load_adapted_bioemu_nodes is public_load_adapted_bioemu_nodes
    assert collector.DatasetBioEmuConfig is PublicDatasetBioEmuConfig
    assert collector.run_dataset_bioemu_experiment is public_run_dataset_bioemu_experiment
    assert collector.write_run_outputs is public_write_run_outputs
