"""Build quality-controlled experimental protein conformation datasets.

The package identifies candidate endpoint pairs and preserves the evidence used
to select them. It does not label any pair as a validated physical transition.
"""

from protein_conformer_dataset.config import DatasetConfig, QCConfig, RankingConfig
from protein_conformer_dataset.models import (
    EndpointPair,
    PairQualityReport,
    PairStatus,
    ResidueMapping,
    StructureRecord,
)
from protein_conformer_dataset.pipeline import DatasetPipeline, RunSummary

__all__ = [
    "DatasetConfig",
    "DatasetPipeline",
    "EndpointPair",
    "PairQualityReport",
    "PairStatus",
    "QCConfig",
    "RankingConfig",
    "ResidueMapping",
    "RunSummary",
    "StructureRecord",
]
