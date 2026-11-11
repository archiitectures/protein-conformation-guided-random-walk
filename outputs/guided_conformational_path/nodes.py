"""Core data structures for protein conformations.

This MVP represents each conformation with one coordinate per residue: the
alpha-carbon position. That keeps the graph prototype lightweight and easy to
test, but it also means validity and clash checks are coarse C-alpha checks
rather than full atomistic validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


VALID_SOURCES = {
    "A_endpoint",
    "B_endpoint",
    "A_generated",
    "B_generated",
    "external",
}


@dataclass(slots=True)
class ConformationNode:
    """A C-alpha protein conformation used as a graph node.

    Attributes:
        node_id: Unique identifier used as the NetworkX node key.
        source: Provenance label. Valid values distinguish endpoints, generated
            nodes from each endpoint, and externally supplied structures.
        coords: NumPy array with shape ``(L, 3)`` where ``L`` is the number of
            residues represented by C-alpha coordinates.
        metadata: Optional free-form provenance/debug information.
        file_path: Optional path to the source PDB/mmCIF-like file.
    """

    node_id: str
    source: str
    coords: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    file_path: str | Path | None = None

    def __post_init__(self) -> None:
        """Validate source labels and normalize coordinates to float arrays."""
        if self.source not in VALID_SOURCES:
            allowed = ", ".join(sorted(VALID_SOURCES))
            raise ValueError(f"Invalid source '{self.source}'. Expected one of: {allowed}")

        coords = np.asarray(self.coords, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(
                f"Coordinates for node '{self.node_id}' must have shape (L, 3); "
                f"got {coords.shape}"
            )
        self.coords = coords

    @property
    def n_residues(self) -> int:
        """Return the number of represented residues/C-alpha coordinates."""
        return int(self.coords.shape[0])
