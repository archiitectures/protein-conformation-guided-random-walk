"""Edge scoring functions for conformation graph construction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from scipy.spatial.distance import jensenshannon, pdist

from guided_conformational_path.nodes import ConformationNode


class EdgeScorer(ABC):
    """Abstract interface for edge-cost scoring between conformation nodes.

    Scores are interpreted as edge costs in the graph: lower values imply a more
    plausible or shorter transition between two candidate conformations.
    """

    @abstractmethod
    def score(self, node_i: ConformationNode, node_j: ConformationNode) -> float:
        """Return a non-negative edge cost; lower means more plausible."""


class RMSDScorer(EdgeScorer):
    """C-alpha RMSD scorer using Kabsch alignment.

    RMSD is computed after optimal rigid-body superposition, so global
    translation and rotation do not affect the score. The two nodes must contain
    the same number of C-alpha coordinates in the same residue order.
    """

    def score(self, node_i: ConformationNode, node_j: ConformationNode) -> float:
        """Return Kabsch-aligned RMSD between two nodes."""
        return kabsch_rmsd(node_i.coords, node_j.coords)


class DistanceHistogramScorer(EdgeScorer):
    """Compare distributions of pairwise C-alpha distances.

    This scorer compares global distance distributions rather than residue-wise
    correspondences. It can be useful as a coarse shape similarity measure, but
    it discards which residue pair produced each distance.
    """

    def __init__(
        self,
        bins: int = 32,
        distance_range: tuple[float, float] = (0.0, 80.0),
        metric: Literal["l2", "js"] = "l2",
    ) -> None:
        """Create a distance-histogram scorer.

        Args:
            bins: Number of histogram bins.
            distance_range: Inclusive distance span passed to NumPy histogram.
            metric: ``"l2"`` for Euclidean histogram distance or ``"js"`` for
                Jensen-Shannon distance.
        """
        if metric not in {"l2", "js"}:
            raise ValueError("metric must be 'l2' or 'js'")
        self.bins = bins
        self.distance_range = distance_range
        self.metric = metric

    def score(self, node_i: ConformationNode, node_j: ConformationNode) -> float:
        """Return distance between pairwise distance histograms."""
        hist_i = distance_histogram(node_i.coords, self.bins, self.distance_range)
        hist_j = distance_histogram(node_j.coords, self.bins, self.distance_range)
        if self.metric == "l2":
            return float(np.linalg.norm(hist_i - hist_j))
        return float(jensenshannon(hist_i, hist_j, base=2.0))


def kabsch_rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Compute RMSD after optimal rigid-body alignment.

    The implementation centers both coordinate sets, computes the covariance
    matrix, obtains the optimal rotation via singular value decomposition, and
    corrects improper rotations before calculating RMSD.
    """
    a = _validated_coords(coords_a, "coords_a")
    b = _validated_coords(coords_b, "coords_b")
    if a.shape != b.shape:
        raise ValueError(f"Coordinate shapes must match for RMSD; got {a.shape} and {b.shape}")

    a_centered = a - a.mean(axis=0)
    b_centered = b - b.mean(axis=0)
    covariance = a_centered.T @ b_centered
    u_mat, _, vt_mat = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(u_mat @ vt_mat))
    rotation = u_mat @ correction @ vt_mat
    aligned = a_centered @ rotation
    diff = aligned - b_centered
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def distance_histogram(
    coords: np.ndarray,
    bins: int = 32,
    distance_range: tuple[float, float] = (0.0, 80.0),
) -> np.ndarray:
    """Return a normalized histogram of all pairwise C-alpha distances.

    Pairwise distances are computed with ``scipy.spatial.distance.pdist`` to
    avoid constructing an explicit ``L x L x 3`` broadcasted difference array.
    """
    array = _validated_coords(coords, "coords")
    if array.shape[0] < 2:
        raise ValueError("At least two coordinates are required for distance histograms")
    distances = pdist(array, metric="euclidean")
    hist, _ = np.histogram(distances, bins=bins, range=distance_range, density=False)
    hist = hist.astype(float)
    total = hist.sum()
    if total == 0:
        return np.full(bins, 1.0 / bins)
    return hist / total


def _validated_coords(coords: np.ndarray, name: str) -> np.ndarray:
    """Normalize and validate a C-alpha coordinate array."""
    array = np.asarray(coords, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (L, 3); got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array
