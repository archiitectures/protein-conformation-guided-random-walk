"""Tests for conformation validity checks."""

from __future__ import annotations

import numpy as np

from guided_conformational_path.nodes import ConformationNode
from guided_conformational_path.validity import check_node_validity


def test_validity_rejects_nan_coordinates() -> None:
    node = ConformationNode("bad", "external", np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]]))
    result = check_node_validity(node)

    assert not result.valid
    assert result.score > 0.0
    assert result.messages


def test_validity_flags_unreasonable_bond_distance() -> None:
    node = ConformationNode(
        "stretched",
        "external",
        np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [23.8, 0.0, 0.0]]),
    )
    result = check_node_validity(node)

    assert not result.valid
    assert result.score > 0.0
    assert any("consecutive C-alpha" in message for message in result.messages)
