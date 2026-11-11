"""Configuration models for dataset construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(slots=True)
class QCConfig:
    """Thresholds that determine whether an endpoint pair is usable.

    Environmental differences that are not hard exclusions remain visible as
    manual-review warnings. They are never silently interpreted as transitions.
    """

    minimum_sequence_identity: float = 0.98
    minimum_aligned_coverage: float = 0.85
    minimum_common_ca_count: int = 40
    maximum_length_difference_fraction: float = 0.10
    maximum_resolution_angstrom: float = 4.0
    exclude_engineered_mutations: bool = False
    exclude_nmr: bool = False
    exclude_multimeric_changes: bool = False
    duplicate_rmsd_angstrom: float = 1.0
    missing_pattern_difference_threshold: float = 0.20

    def __post_init__(self) -> None:
        """Reject nonsensical fractions and coordinate thresholds."""
        for name in (
            "minimum_sequence_identity",
            "minimum_aligned_coverage",
            "maximum_length_difference_fraction",
            "missing_pattern_difference_threshold",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1; got {value}")
        if self.minimum_common_ca_count < 1:
            raise ValueError("minimum_common_ca_count must be at least 1")
        if self.maximum_resolution_angstrom <= 0 or self.duplicate_rmsd_angstrom < 0:
            raise ValueError("Resolution must be positive and duplicate RMSD must be non-negative")

    @classmethod
    def from_mode(cls, mode: Literal["permissive", "strict"]) -> "QCConfig":
        """Return the named QC profile."""
        if mode == "permissive":
            return cls()
        if mode == "strict":
            return cls(
                exclude_engineered_mutations=True,
                exclude_nmr=True,
                exclude_multimeric_changes=True,
            )
        raise ValueError(f"Unknown QC mode: {mode!r}")


@dataclass(slots=True)
class RankingConfig:
    """Weights for structural endpoint candidate ranking, not physics."""

    rmsd_weight: float = 1.0
    coverage_weight: float = 0.75
    identity_weight: float = 0.75
    resolution_weight: float = 0.25
    mutation_penalty: float = 0.25
    missing_penalty: float = 0.5


@dataclass(slots=True)
class DatasetConfig:
    """Top-level configuration for a reproducible dataset run."""

    output_dir: Path = Path("data/dataset")
    cache_dir: Path = Path("data/cache")
    top_k_pairs: int = 3
    max_structures_per_protein: int = 50
    request_timeout_seconds: float = 30.0
    request_rate_seconds: float = 0.1
    request_retries: int = 3
    resume: bool = False
    force: bool = False
    qc_mode: Literal["permissive", "strict"] = "permissive"
    qc: QCConfig = field(default_factory=QCConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)

    def __post_init__(self) -> None:
        """Normalize paths and reject invalid limits."""
        self.output_dir = Path(self.output_dir)
        self.cache_dir = Path(self.cache_dir)
        if self.top_k_pairs < 1:
            raise ValueError("top_k_pairs must be at least 1")
        if self.max_structures_per_protein < 2:
            raise ValueError("max_structures_per_protein must be at least 2")
        if self.qc_mode == "strict":
            self.qc.exclude_engineered_mutations = True
            self.qc.exclude_nmr = True
            self.qc.exclude_multimeric_changes = True

    @classmethod
    def with_qc_mode(cls, *, qc_mode: Literal["permissive", "strict"], **kwargs: Any) -> "DatasetConfig":
        """Construct a config with the selected QC profile."""
        return cls(qc_mode=qc_mode, qc=QCConfig.from_mode(qc_mode), **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-ready representation."""
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        data["cache_dir"] = str(self.cache_dir)
        return data
