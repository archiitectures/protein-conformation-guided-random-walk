"""Node generator interfaces and MVP implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from guided_conformational_path.io import read_pdb_ca
from guided_conformational_path.nodes import ConformationNode
from guided_conformational_path.validity import ValidityResult, check_node_validity


class NodeGenerator(ABC):
    """Abstract interface for proposing conformation nodes from an endpoint.

    A generator receives one endpoint conformation and returns zero or more
    candidate conformations represented as ``ConformationNode`` objects. Future
    adapters for BioEmu, EigenFold, or other ensemble models should implement
    this interface so the graph and path-search layers do not depend on a
    particular generation backend.
    """

    @abstractmethod
    def generate_from_endpoint(
        self,
        endpoint: ConformationNode,
        n_samples: int,
        **kwargs: Any,
    ) -> list[ConformationNode]:
        """Generate candidate conformations near an endpoint.

        Implementations may return fewer than ``n_samples`` if validity filters,
        missing input files, or model-side failures prevent all requested samples
        from being produced. They should record that behavior in node metadata or
        generator-level summary fields where possible.
        """


class DummyPerturbationGenerator(NodeGenerator):
    """Generate synthetic conformations by Gaussian coordinate perturbation.

    This is a pipeline test generator, not a physical protein sampler. It adds
    independent Gaussian noise to each C-alpha coordinate and can reject samples
    that fail the lightweight checks in ``validity.py``. The rejection filter is
    intended to avoid obviously broken nodes, such as non-neighbor C-alpha pairs
    occupying nearly the same space.
    """

    def __init__(
        self,
        noise_scale: float = 0.75,
        seed: int | None = 7,
        enforce_validity: bool = True,
        max_attempts_per_sample: int = 10,
        min_bond_distance: float = 2.5,
        max_bond_distance: float = 4.5,
        clash_distance: float | None = 2.0,
    ) -> None:
        """Create a dummy Gaussian generator.

        Args:
            noise_scale: Standard deviation of the independent Gaussian noise
                added to each coordinate component, in Angstrom-like units.
            seed: Random seed for reproducible demo/test output. ``None`` uses
                NumPy's non-deterministic default seeding.
            enforce_validity: If true, reject generated samples that fail
                ``check_node_validity``.
            max_attempts_per_sample: Attempt budget is
                ``n_samples * max_attempts_per_sample``. If the budget is
                exhausted, the generator returns the valid subset instead of
                raising.
            min_bond_distance: Lower bound for consecutive C-alpha distances.
            max_bond_distance: Upper bound for consecutive C-alpha distances.
            clash_distance: Optional non-neighbor C-alpha clash threshold.
        """
        if noise_scale < 0:
            raise ValueError("noise_scale must be non-negative")
        if max_attempts_per_sample <= 0:
            raise ValueError("max_attempts_per_sample must be positive")
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng(seed)
        self.enforce_validity = enforce_validity
        self.max_attempts_per_sample = max_attempts_per_sample
        self.min_bond_distance = min_bond_distance
        self.max_bond_distance = max_bond_distance
        self.clash_distance = clash_distance
        self.last_generation_summary: dict[str, Any] = {}

    def generate_from_endpoint(
        self,
        endpoint: ConformationNode,
        n_samples: int,
        **kwargs: Any,
    ) -> list[ConformationNode]:
        """Return validity-filtered noisy copies of endpoint coordinates.

        The requested ``n_samples`` is a target, not a guarantee. With validity
        enforcement enabled, the generator proposes up to
        ``n_samples * max_attempts_per_sample`` noisy conformations and returns
        only those that pass finite-coordinate, bond-distance, and coarse
        C-alpha clash checks. If too many proposals fail, fewer nodes are
        returned and ``last_generation_summary`` records the accepted/rejected
        counts and the most recent rejection messages.
        """
        if n_samples < 0:
            raise ValueError("n_samples must be non-negative")
        source = kwargs.get("source") or _generated_source_for(endpoint.source)
        nodes: list[ConformationNode] = []
        rejected_count = 0
        last_rejection_messages: list[str] = []
        max_attempts = n_samples * self.max_attempts_per_sample

        for attempt_idx in range(max_attempts):
            noise = self.rng.normal(
                loc=0.0,
                scale=self.noise_scale,
                size=endpoint.coords.shape,
            )
            candidate = ConformationNode(
                node_id=f"{endpoint.node_id}_gen_{len(nodes):03d}",
                source=source,
                coords=endpoint.coords + noise,
                metadata={},
            )
            validity = self._check_candidate(candidate)
            if self.enforce_validity and not validity.valid:
                rejected_count += 1
                last_rejection_messages = validity.messages
                continue

            nodes.append(
                ConformationNode(
                    node_id=candidate.node_id,
                    source=source,
                    coords=candidate.coords,
                    metadata={
                        "generator": self.__class__.__name__,
                        "parent": endpoint.node_id,
                        "noise_scale": self.noise_scale,
                        "validity_enforced": self.enforce_validity,
                        "attempt_index": attempt_idx,
                        "accepted_count": len(nodes) + 1,
                        "rejected_count_so_far": rejected_count,
                        "validity_score": validity.score,
                        "validity_messages": validity.messages,
                        "validity_settings": self._validity_settings(),
                    },
                )
            )
            if len(nodes) == n_samples:
                break

        self.last_generation_summary = {
            "requested": n_samples,
            "returned": len(nodes),
            "rejected": rejected_count,
            "attempted": len(nodes) + rejected_count,
            "max_attempts": max_attempts,
            "validity_enforced": self.enforce_validity,
            "last_rejection_messages": last_rejection_messages,
            "validity_settings": self._validity_settings(),
        }
        return nodes

    def _check_candidate(self, node: ConformationNode) -> ValidityResult:
        """Run the same lightweight validity checks used by graph penalties."""
        return check_node_validity(
            node,
            min_bond_distance=self.min_bond_distance,
            max_bond_distance=self.max_bond_distance,
            clash_distance=self.clash_distance,
        )

    def _validity_settings(self) -> dict[str, float | bool | None]:
        """Return validity settings for metadata/debug summaries."""
        return {
            "min_bond_distance": self.min_bond_distance,
            "max_bond_distance": self.max_bond_distance,
            "clash_distance": self.clash_distance,
        }


class PrecomputedNodeGenerator(NodeGenerator):
    """Load candidate conformations from a directory of PDB files.

    This generator is useful when another tool has already produced candidate
    structures. It does not infer conformations itself; it reads up to
    ``n_samples`` ``*.pdb`` files, extracts C-alpha coordinates, and wraps them
    as ``ConformationNode`` objects with provenance metadata.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def generate_from_endpoint(
        self,
        endpoint: ConformationNode,
        n_samples: int,
        **kwargs: Any,
    ) -> list[ConformationNode]:
        """Load up to ``n_samples`` PDB files from the configured directory.

        Files are consumed in sorted path order for reproducibility. If fewer PDB
        files exist than requested, the available subset is returned.
        """
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
    """Placeholder adapter for future BioEmu inference.

    The future implementation should call BioEmu, convert generated structures
    into C-alpha coordinate arrays, and return ``ConformationNode`` objects. The
    rest of the pipeline should not need to know that BioEmu produced the nodes.
    """

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
    """Placeholder adapter for future EigenFold inference.

    The future implementation should mirror the BioEmu adapter shape: run
    EigenFold externally, convert outputs into C-alpha coordinate arrays, and
    return ``ConformationNode`` instances.
    """

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
    """Map endpoint source labels to generated-node source labels."""
    if endpoint_source == "A_endpoint":
        return "A_generated"
    if endpoint_source == "B_endpoint":
        return "B_generated"
    return "external"
