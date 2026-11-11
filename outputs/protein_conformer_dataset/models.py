"""Serializable data models used throughout the dataset pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class PairStatus(str, Enum):
    """Exclusive endpoint-pair review status."""

    ACCEPTED = "accepted"
    MANUAL_REVIEW = "manual_review"
    REJECTED = "rejected"


@dataclass(slots=True)
class StructureRecord:
    """Metadata for one selected experimental polymer entity and chain."""

    pdb_id: str
    entity_id: str | None
    auth_chain_id: str
    label_chain_id: str | None = None
    uniprot_id: str | None = None
    protein_name: str | None = None
    organism: str | None = None
    experimental_method: str | None = None
    resolution: float | None = None
    deposition_date: str | None = None
    sequence: str = ""
    sequence_length: int = 0
    observed_residue_count: int = 0
    ligand_ids: list[str] = field(default_factory=list)
    metal_ids: list[str] = field(default_factory=list)
    nucleotide_ids: list[str] = field(default_factory=list)
    partner_chain_ids: list[str] = field(default_factory=list)
    nucleic_acid_partner_chain_ids: list[str] = field(default_factory=list)
    mutation_annotations: list[str] = field(default_factory=list)
    file_path: str | None = None
    model_count: int = 1
    alternate_locations_present: bool = False
    oligomeric_state: str | None = None
    biological_assembly_id: str | None = None
    membrane_associated: bool | None = None
    ph: float | None = None
    temperature: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize PDB and accession identifiers."""
        self.pdb_id = self.pdb_id.upper()
        if self.uniprot_id:
            self.uniprot_id = self.uniprot_id.upper()

    @property
    def structure_id(self) -> str:
        """Return a stable PDB/entity/chain identifier."""
        entity = self.entity_id or "_"
        return f"{self.pdb_id}_{entity}_{self.auth_chain_id}"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StructureRecord":
        """Restore a record from serialized data."""
        return cls(**data)


@dataclass(slots=True)
class ParsedResidue:
    """One structure-sequence position and its optional observed C-alpha."""

    index: int
    residue_name: str
    one_letter: str
    auth_seq_id: str | int | None
    label_seq_id: int | None
    insertion_code: str = ""
    ca_coord: np.ndarray | None = None
    occupancy: float | None = None
    altloc: str | None = None
    alternate_locations_present: bool = False
    uniprot_position: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dictionary."""
        data = asdict(self)
        data["ca_coord"] = None if self.ca_coord is None else self.ca_coord.tolist()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParsedResidue":
        """Restore a residue from serialized data."""
        values = dict(data)
        if values.get("ca_coord") is not None:
            values["ca_coord"] = np.asarray(values["ca_coord"], dtype=float)
        return cls(**values)


@dataclass(slots=True)
class ParsedChain:
    """A polymer chain represented in full sequence order."""

    record: StructureRecord
    residues: list[ParsedResidue]

    @property
    def sequence(self) -> str:
        """Return the complete parsed construct sequence."""
        return "".join(residue.one_letter for residue in self.residues)

    @property
    def ca_count(self) -> int:
        """Return the number of residues with observed C-alpha atoms."""
        return sum(residue.ca_coord is not None for residue in self.residues)

    def to_dict(self) -> dict[str, Any]:
        """Return a checkpoint-ready dictionary."""
        return {"record": self.record.to_dict(), "residues": [r.to_dict() for r in self.residues]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParsedChain":
        """Restore a parsed chain from serialized data."""
        return cls(
            record=StructureRecord.from_dict(data["record"]),
            residues=[ParsedResidue.from_dict(item) for item in data["residues"]],
        )


@dataclass(slots=True)
class ResidueMapping:
    """One sequence-aligned residue pair and coordinate availability."""

    index_a: int
    index_b: int
    auth_seq_id_a: str | int | None
    auth_seq_id_b: str | int | None
    residue_name_a: str
    residue_name_b: str
    ca_present_a: bool
    ca_present_b: bool
    uniprot_position_a: int | None = None
    uniprot_position_b: int | None = None
    chain_a: str | None = None
    chain_b: str | None = None


@dataclass(slots=True)
class PairQualityReport:
    """Sequence, structure, and environment evidence for an endpoint pair."""

    sequence_identity: float
    aligned_coverage_a: float
    aligned_coverage_b: float
    common_ca_count: int
    raw_ca_rmsd: float | None
    ca_rmsd: float | None
    length_difference: int
    mutation_count: int
    missing_fraction_a: float
    missing_fraction_b: float
    same_uniprot: bool
    same_organism: bool | None
    oligomeric_state_match: bool | None
    warnings: list[str]
    rejection_reasons: list[str]
    status: PairStatus

    @property
    def accepted(self) -> bool:
        """Return whether the pair passed without mandatory manual review."""
        return self.status is PairStatus.ACCEPTED

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dictionary with explicit status and boolean."""
        data = asdict(self)
        data["status"] = self.status.value
        data["accepted"] = self.accepted
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PairQualityReport":
        """Restore a report from serialized data."""
        values = dict(data)
        values.pop("accepted", None)
        values["status"] = PairStatus(values["status"])
        return cls(**values)


@dataclass(slots=True)
class EndpointPair:
    """Two experimental structures plus residue correspondence and QC."""

    protein_id: str
    uniprot_id: str | None
    structure_a: StructureRecord
    structure_b: StructureRecord
    quality: PairQualityReport
    residue_mapping: list[ResidueMapping]
    coords_a: np.ndarray
    coords_b: np.ndarray
    aligned_coords_a: np.ndarray
    aligned_coords_b: np.ndarray
    candidate_score: float = 0.0
    state_a: str | None = None
    state_b: str | None = None

    @property
    def pair_id(self) -> str:
        """Return a filesystem-safe stable endpoint identifier."""
        return (
            f"{self.structure_a.pdb_id}_{self.structure_a.auth_chain_id}__"
            f"{self.structure_b.pdb_id}_{self.structure_b.auth_chain_id}"
        )

    def to_dict(self, include_coordinates: bool = True) -> dict[str, Any]:
        """Return a JSON/checkpoint representation."""
        data: dict[str, Any] = {
            "protein_id": self.protein_id,
            "uniprot_id": self.uniprot_id,
            "structure_a": self.structure_a.to_dict(),
            "structure_b": self.structure_b.to_dict(),
            "quality": self.quality.to_dict(),
            "residue_mapping": [asdict(item) for item in self.residue_mapping],
            "candidate_score": self.candidate_score,
            "state_a": self.state_a,
            "state_b": self.state_b,
            "pair_id": self.pair_id,
        }
        if include_coordinates:
            data.update(
                coords_a=self.coords_a.tolist(),
                coords_b=self.coords_b.tolist(),
                aligned_coords_a=self.aligned_coords_a.tolist(),
                aligned_coords_b=self.aligned_coords_b.tolist(),
            )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EndpointPair":
        """Restore a pair from a checkpoint payload."""
        return cls(
            protein_id=data["protein_id"],
            uniprot_id=data.get("uniprot_id"),
            structure_a=StructureRecord.from_dict(data["structure_a"]),
            structure_b=StructureRecord.from_dict(data["structure_b"]),
            quality=PairQualityReport.from_dict(data["quality"]),
            residue_mapping=[ResidueMapping(**item) for item in data["residue_mapping"]],
            coords_a=np.asarray(data["coords_a"], dtype=float),
            coords_b=np.asarray(data["coords_b"], dtype=float),
            aligned_coords_a=np.asarray(data["aligned_coords_a"], dtype=float),
            aligned_coords_b=np.asarray(data["aligned_coords_b"], dtype=float),
            candidate_score=float(data.get("candidate_score", 0.0)),
            state_a=data.get("state_a"),
            state_b=data.get("state_b"),
        )
