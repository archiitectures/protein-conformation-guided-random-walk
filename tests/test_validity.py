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


def test_validity_flags_non_neighbor_ca_clash() -> None:
    node = ConformationNode(
        "clash",
        "external",
        np.array(
            [
                [0.0, 0.0, 0.0],
                [3.8, 0.0, 0.0],
                [0.2, 0.0, 0.0],
            ]
        ),
    )
    result = check_node_validity(node, clash_distance=2.0)

    assert not result.valid
    assert any("clash threshold" in message for message in result.messages)


def test_validity_does_not_treat_bonded_neighbors_as_clashes() -> None:
    node = ConformationNode(
        "neighbors",
        "external",
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
    )
    result = check_node_validity(
        node,
        min_bond_distance=0.5,
        max_bond_distance=4.5,
        clash_distance=2.0,
    )

    assert result.valid
    assert result.messages == []


def test_validity_skips_bond_check_across_missing_residue_positions() -> None:
    """Coordinate neighbors separated by a sequence gap are not bonded."""
    node = ConformationNode(
        "gapped",
        "external",
        np.array(
            [
                [0.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
                [23.8, 0.0, 0.0],
            ]
        ),
        metadata={"residue_positions": [1, 3, 4]},
    )

    result = check_node_validity(node, clash_distance=None)

    assert result.valid
    assert result.messages == []


def test_validity_checks_clash_across_adjacent_rows_with_sequence_gap() -> None:
    """Rows separated by missing residues remain eligible for clash checks."""
    node = ConformationNode(
        "gapped_clash",
        "external",
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [4.3, 0.0, 0.0],
            ]
        ),
        metadata={"residue_positions": [1, 3, 4]},
    )

    result = check_node_validity(
        node,
        min_bond_distance=2.5,
        max_bond_distance=4.5,
        clash_distance=2.0,
    )

    assert not result.valid
    assert any("clash threshold" in message for message in result.messages)
