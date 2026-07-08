"""Lightweight physical validity checks for C-alpha conformations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from guided_conformational_path.nodes import ConformationNode


@dataclass(slots=True)
class ValidityResult:
    """Result of simple conformation validity checks."""

    valid: bool
    score: float
    messages: list[str]


def check_node_validity(
    node: ConformationNode,
    min_bond_distance: float = 2.5,
    max_bond_distance: float = 4.5,
    clash_distance: float | None = 2.0,
) -> ValidityResult:
    """Check finite coordinates, C-alpha bond distances, and coarse clashes."""
    coords = np.asarray(node.coords, dtype=float)
    messages: list[str] = []
    penalty = 0.0

    if coords.ndim != 2 or coords.shape[1] != 3:
        return ValidityResult(False, 1_000.0, [f"Invalid coordinate shape: {coords.shape}"])

    if not np.all(np.isfinite(coords)):
        return ValidityResult(False, 1_000.0, ["Coordinates contain NaN or infinite values"])

    if coords.shape[0] >= 2:
        bond_distances = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        low = bond_distances < min_bond_distance
        high = bond_distances > max_bond_distance
        if np.any(low) or np.any(high):
            count = int(np.count_nonzero(low) + np.count_nonzero(high))
            messages.append(f"{count} consecutive C-alpha distances outside expected range")
            penalty += float(np.sum(np.maximum(min_bond_distance - bond_distances, 0.0)))
            penalty += float(np.sum(np.maximum(bond_distances - max_bond_distance, 0.0)))

    if clash_distance is not None and coords.shape[0] >= 3:
        deltas = coords[:, None, :] - coords[None, :, :]
        distances = np.sqrt(np.sum(deltas * deltas, axis=-1))
        mask = np.triu(np.ones(distances.shape, dtype=bool), k=2)
        clashes = distances[mask] < clash_distance
        if np.any(clashes):
            count = int(np.count_nonzero(clashes))
            messages.append(f"{count} non-neighbor C-alpha distances below clash threshold")
            penalty += float(count)

    return ValidityResult(valid=penalty == 0.0, score=penalty, messages=messages)
