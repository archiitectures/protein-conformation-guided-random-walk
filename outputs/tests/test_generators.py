"""Tests for node generators."""

from __future__ import annotations

import numpy as np

from guided_conformational_path.generators import DummyPerturbationGenerator
from guided_conformational_path.nodes import ConformationNode


def test_dummy_generator_count_shape_and_source() -> None:
    endpoint = ConformationNode("A", "A_endpoint", _valid_chain(6))
    nodes = DummyPerturbationGenerator(noise_scale=0.05, seed=3).generate_from_endpoint(endpoint, 4)

    assert len(nodes) == 4
    assert all(node.coords.shape == endpoint.coords.shape for node in nodes)
    assert all(node.source == "A_generated" for node in nodes)
    assert all(node.metadata["validity_enforced"] for node in nodes)


def test_dummy_generator_filters_invalid_clashing_samples() -> None:
    endpoint = ConformationNode(
        "A",
        "A_endpoint",
        np.array(
            [
                [0.0, 0.0, 0.0],
                [3.8, 0.0, 0.0],
                [0.1, 0.0, 0.0],
            ]
        ),
    )
    generator = DummyPerturbationGenerator(
        noise_scale=0.0,
        seed=3,
        max_attempts_per_sample=2,
        clash_distance=2.0,
    )

    nodes = generator.generate_from_endpoint(endpoint, 3)

    assert nodes == []
    assert generator.last_generation_summary["requested"] == 3
    assert generator.last_generation_summary["returned"] == 0
    assert generator.last_generation_summary["rejected"] == 6
    assert generator.last_generation_summary["last_rejection_messages"]


def test_dummy_generator_can_disable_validity_filter() -> None:
    endpoint = ConformationNode("A", "A_endpoint", np.zeros((3, 3)))
    generator = DummyPerturbationGenerator(noise_scale=0.0, enforce_validity=False)

    nodes = generator.generate_from_endpoint(endpoint, 2)

    assert len(nodes) == 2
    assert generator.last_generation_summary["validity_enforced"] is False


def _valid_chain(n_residues: int) -> np.ndarray:
    residue_idx = np.arange(n_residues, dtype=float)
    return np.column_stack([residue_idx * 3.8, np.zeros(n_residues), np.zeros(n_residues)])
