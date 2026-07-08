"""Core data structures for protein conformations."""

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
    """A C-alpha protein conformation used as a graph node."""

    node_id: str
    source: str
    coords: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    file_path: str | Path | None = None

    def __post_init__(self) -> None:
        """Validate and normalize node fields."""
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
        """Return the number of C-alpha coordinates."""
        return int(self.coords.shape[0])
