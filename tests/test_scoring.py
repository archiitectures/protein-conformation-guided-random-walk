"""Tests for conformation edge scoring."""

from __future__ import annotations

import numpy as np
import pytest

from guided_conformational_path.nodes import ConformationNode
from guided_conformational_path.scoring import (
    DistanceHistogramScorer,
    RMSDScorer,
    distance_histogram,
    kabsch_rmsd,
)


def test_kabsch_rmsd_zero_for_rotated_translated_coordinates() -> None:
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transformed = coords @ rotation + np.array([10.0, -4.0, 2.5])
    assert kabsch_rmsd(coords, transformed) == pytest.approx(0.0, abs=1e-12)


def test_rmsd_scorer_rejects_mismatched_lengths() -> None:
    node_a = ConformationNode("a", "external", np.zeros((3, 3)))
    node_b = ConformationNode("b", "external", np.zeros((4, 3)))
    with pytest.raises(ValueError, match="Coordinate shapes must match"):
        RMSDScorer().score(node_a, node_b)


def test_distance_histogram_score_is_finite_and_nonnegative() -> None:
    node_a = ConformationNode("a", "external", np.arange(15, dtype=float).reshape(5, 3))
    node_b = ConformationNode("b", "external", np.arange(15, dtype=float).reshape(5, 3) + 0.5)
    score = DistanceHistogramScorer(metric="js").score(node_a, node_b)
    assert np.isfinite(score)
    assert score >= 0.0


def test_distance_histogram_is_normalized() -> None:
    coords = np.arange(18, dtype=float).reshape(6, 3)
    hist = distance_histogram(coords, bins=8, distance_range=(0.0, 40.0))

    assert hist.shape == (8,)
    assert hist.sum() == pytest.approx(1.0)
