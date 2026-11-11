"""Lightweight physical validity checks for C-alpha conformations.

The current data model stores only C-alpha coordinates, so these checks are
coarse filters rather than full molecular validation. In particular, the clash
check is a C-alpha-only proxy that catches severe coordinate overlaps; it is not
equivalent to full atom-atom steric clash detection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from guided_conformational_path.nodes import ConformationNode


@dataclass(slots=True)
class ValidityResult:
    """Result of simple conformation validity checks.

    Attributes:
        valid: True when no penalty-producing checks fired.
        score: Additive penalty. Larger values indicate more severe violations.
        messages: Human-readable reasons for penalties, suitable for metadata
            and debugging.
    """

    valid: bool
    score: float
    messages: list[str]


def check_node_validity(
    node: ConformationNode,
    min_bond_distance: float = 2.5,
    max_bond_distance: float = 4.5,
    clash_distance: float | None = 2.0,
) -> ValidityResult:
    """Check finite coordinates, C-alpha bond distances, and coarse clashes.

    Args:
        node: Conformation to validate.
        min_bond_distance: Minimum expected distance between consecutive
            C-alpha atoms.
        max_bond_distance: Maximum expected distance between consecutive
            C-alpha atoms.
        clash_distance: If provided, non-neighbor C-alpha pairs closer than this
            threshold are counted as clash-like overlaps.

    Returns:
        A ``ValidityResult`` containing a boolean decision, additive penalty,
        and explanatory messages. Consecutive residues are excluded from the
        clash-like check because bonded neighbors are expected to be close.
    """
    coords = np.asarray(node.coords, dtype=float)
    messages: list[str] = []
    penalty = 0.0

    if coords.ndim != 2 or coords.shape[1] != 3:
        return ValidityResult(False, 1_000.0, [f"Invalid coordinate shape: {coords.shape}"])

    if not np.all(np.isfinite(coords)):
        return ValidityResult(False, 1_000.0, ["Coordinates contain NaN or infinite values"])

    residue_positions = _residue_positions(node, coords.shape[0])

    if coords.shape[0] >= 2:
        row_distances = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        if residue_positions is None:
            bond_distances = row_distances
        else:
            consecutive = np.diff(residue_positions) == 1
            bond_distances = row_distances[consecutive]
        low = bond_distances < min_bond_distance
        high = bond_distances > max_bond_distance
        if np.any(low) or np.any(high):
            count = int(np.count_nonzero(low) + np.count_nonzero(high))
            messages.append(f"{count} consecutive C-alpha distances outside expected range")
            penalty += float(np.sum(np.maximum(min_bond_distance - bond_distances, 0.0)))
            penalty += float(np.sum(np.maximum(bond_distances - max_bond_distance, 0.0)))

    if clash_distance is not None and coords.shape[0] >= 3:
        clashes = non_neighbor_ca_clashes(
            coords,
            clash_distance,
            residue_positions=residue_positions,
        )
        if np.any(clashes):
            count = int(np.count_nonzero(clashes))
            messages.append(f"{count} non-neighbor C-alpha distances below clash threshold")
            penalty += float(count)

    return ValidityResult(valid=penalty == 0.0, score=penalty, messages=messages)


def non_neighbor_ca_clashes(
    coords: np.ndarray,
    clash_distance: float,
    residue_positions: np.ndarray | None = None,
) -> np.ndarray:
    """Return a boolean vector for non-neighbor C-alpha clash-like distances.

    The returned array corresponds to all residue pairs separated by at least two
    sequence positions. It is intentionally C-alpha-only because full atom
    coordinates are not represented in this MVP.
    """
    array = np.asarray(coords, dtype=float)
    deltas = array[:, None, :] - array[None, :, :]
    distances = np.sqrt(np.sum(deltas * deltas, axis=-1))
    if residue_positions is None:
        mask = np.triu(np.ones(distances.shape, dtype=bool), k=2)
    else:
        positions = np.asarray(residue_positions, dtype=int)
        if positions.shape != (array.shape[0],):
            raise ValueError(
                "residue_positions must have one entry per coordinate; "
                f"got {positions.shape} for {array.shape[0]} coordinates"
            )
        sequence_separation = np.abs(positions[:, None] - positions[None, :])
        mask = np.triu(sequence_separation > 1, k=1)
    return distances[mask] < clash_distance


def _residue_positions(node: ConformationNode, coordinate_count: int) -> np.ndarray | None:
    """Return validated one-based sequence positions from node metadata."""
    values = node.metadata.get("residue_positions")
    if values is None:
        return None
    positions = np.asarray(values, dtype=int)
    if positions.shape != (coordinate_count,):
        raise ValueError(
            f"Node {node.node_id!r} has {coordinate_count} coordinates but "
            f"{positions.size} residue positions"
        )
    if np.any(positions < 1) or np.any(np.diff(positions) <= 0):
        raise ValueError(
            f"Node {node.node_id!r} residue_positions must be strictly increasing positive integers"
        )
    return positions
