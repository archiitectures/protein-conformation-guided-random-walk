"""Node generator interfaces and MVP implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from guided_conformational_path.io import read_pdb_ca
from guided_conformational_path.nodes import ConformationNode


class NodeGenerator(ABC):
    """Abstract interface for proposing conformation nodes from an endpoint."""

    @abstractmethod
    def generate_from_endpoint(
        self,
        endpoint: ConformationNode,
        n_samples: int,
        **kwargs: Any,
    ) -> list[ConformationNode]:
        """Generate candidate conformations near an endpoint."""


class DummyPerturbationGenerator(NodeGenerator):
    """Generate synthetic conformations by Gaussian coordinate perturbation."""

    def __init__(self, noise_scale: float = 0.75, seed: int | None = 7) -> None:
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng(seed)

    def generate_from_endpoint(
        self,
        endpoint: ConformationNode,
        n_samples: int,
        **kwargs: Any,
    ) -> list[ConformationNode]:
        """Return noisy copies of the endpoint coordinates."""
        if n_samples < 0:
            raise ValueError("n_samples must be non-negative")
        source = kwargs.get("source") or _generated_source_for(endpoint.source)
        nodes: list[ConformationNode] = []
        for idx in range(n_samples):
            noise = self.rng.normal(
                loc=0.0,
                scale=self.noise_scale,
                size=endpoint.coords.shape,
            )
            nodes.append(
                ConformationNode(
                    node_id=f"{endpoint.node_id}_gen_{idx:03d}",
                    source=source,
                    coords=endpoint.coords + noise,
                    metadata={
                        "generator": self.__class__.__name__,
                        "parent": endpoint.node_id,
                        "noise_scale": self.noise_scale,
                    },
                )
            )
        return nodes


class PrecomputedNodeGenerator(NodeGenerator):
    """Load candidate conformations from a directory of PDB files."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def generate_from_endpoint(
        self,
        endpoint: ConformationNode,
        n_samples: int,
        **kwargs: Any,
    ) -> list[ConformationNode]:
        """Load up to n_samples PDB files from the configured directory."""
        if not self.directory.exists():
            raise FileNotFoundError(f"Precomputed directory does not exist: {self.directory}")
        source = kwargs.get("source") or _generated_source_for(endpoint.source)
        files = sorted(self.directory.glob("*.pdb"))[:n_samples]
        nodes: list[ConformationNode] = []
        for idx, path in enumerate(files):
            nodes.append(
                ConformationNode(
                    node_id=f"{endpoint.node_id}_precomputed_{idx:03d}",
                    source=source,
                    coords=read_pdb_ca(path),
                    metadata={"generator": self.__class__.__name__, "parent": endpoint.node_id},
                    file_path=path,
                )
            )
        return nodes


class BioEmuNodeGenerator(NodeGenerator):
    """Placeholder adapter for future BioEmu inference."""

    def generate_from_endpoint(
        self,
        endpoint: ConformationNode,
        n_samples: int,
        **kwargs: Any,
    ) -> list[ConformationNode]:
        """Raise until BioEmu integration is added."""
        raise NotImplementedError(
            "BioEmu integration is not implemented in this MVP. "
            "Implement this adapter with the same NodeGenerator interface."
        )


class EigenFoldNodeGenerator(NodeGenerator):
    """Placeholder adapter for future EigenFold inference."""

    def generate_from_endpoint(
        self,
        endpoint: ConformationNode,
        n_samples: int,
        **kwargs: Any,
    ) -> list[ConformationNode]:
        """Raise until EigenFold integration is added."""
        raise NotImplementedError(
            "EigenFold integration is not implemented in this MVP. "
            "Implement this adapter with the same NodeGenerator interface."
        )


def _generated_source_for(endpoint_source: str) -> str:
    if endpoint_source == "A_endpoint":
        return "A_generated"
    if endpoint_source == "B_endpoint":
        return "B_generated"
    return "external"
