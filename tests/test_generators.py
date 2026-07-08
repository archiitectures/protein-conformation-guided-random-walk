"""Tests for node generators."""

from __future__ import annotations

import numpy as np

from guided_conformational_path.generators import DummyPerturbationGenerator
from guided_conformational_path.nodes import ConformationNode


def test_dummy_generator_count_shape_and_source() -> None:
    endpoint = ConformationNode("A", "A_endpoint", np.zeros((6, 3)))
    nodes = DummyPerturbationGenerator(seed=3).generate_from_endpoint(endpoint, 4)

    assert len(nodes) == 4
    assert all(node.coords.shape == endpoint.coords.shape for node in nodes)
    assert all(node.source == "A_generated" for node in nodes)
