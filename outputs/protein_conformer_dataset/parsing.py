"""Robust mmCIF protein-chain parsing with Gemmi."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import gemmi
import numpy as np

from protein_conformer_dataset.models import ParsedChain, ParsedResidue, StructureRecord
from protein_conformer_dataset.sequence import residue_to_one_letter


WATER_NAMES = {"HOH", "DOD", "WAT"}
METAL_NAMES = {
    "ZN", "MG", "MN", "FE", "CA", "NA", "K", "CU", "CO", "NI", "CD", "HG",
}
NUCLEOTIDE_NAMES = {"ATP", "ADP", "AMP", "GTP", "GDP", "GMP", "CTP", "UTP", "NAD", "FAD"}
NUCLEIC_NAMES = {"A", "C", "G", "U", "I", "DA", "DC", "DG", "DT", "DU"}


class StructureParseError(ValueError):
    """Raised when an mmCIF file or requested chain is unusable."""


def parse_mmcif_chain(path: str | Path, record: StructureRecord) -> ParsedChain:
    """Parse one chain in full entity-sequence order.

    The first model supplies coordinates. For NMR entries the total number of
    models is retained in metadata. C-alpha alternate locations are resolved by
    highest occupancy, with blank and then ``A`` preferred when tied.
    """
    source = Path(path)
    if not source.exists() or source.stat().st_size == 0:
        raise StructureParseError(f"mmCIF file is missing or empty: {source}")
    try:
        structure = gemmi.read_structure(str(source))
    except Exception as exc:
        raise StructureParseError(f"Could not parse mmCIF file {source}: {exc}") from exc
    if len(structure) == 0:
        raise StructureParseError(f"mmCIF file contains no models: {source}")
    model = structure[0]
    chain = _find_chain(model, record.auth_chain_id, record.label_chain_id)
    if chain is None:
        available = ", ".join(sorted(item.name for item in model))
        raise StructureParseError(
            f"Chain {record.auth_chain_id!r}/{record.label_chain_id!r} not found in "
            f"{record.pdb_id}; available author chains: {available}"
        )

    observed = _observed_protein_residues(chain)
    if not observed:
        raise StructureParseError(f"No amino-acid residues found in {record.pdb_id} chain {chain.name}")
    full_sequence = _entity_sequence(structure, observed)
    if not full_sequence:
        full_sequence = record.sequence or "".join(item.one_letter for item in observed)
    residues = _merge_full_and_observed(full_sequence, observed)

    ligands, metals, nucleotides = _selected_chain_environment(chain)
    protein_partners, nucleic_partners = _partner_chains(model, chain.name)
    record.auth_chain_id = chain.name
    if observed[0].label_chain_id:
        record.label_chain_id = observed[0].label_chain_id
    record.sequence = "".join(item.one_letter for item in residues)
    record.sequence_length = len(residues)
    record.observed_residue_count = sum(item.ca_coord is not None for item in residues)
    record.file_path = str(source)
    record.model_count = len(structure)
    record.alternate_locations_present = any(item.alternate_locations_present for item in residues)
    record.ligand_ids = sorted(set(record.ligand_ids) | ligands)
    record.metal_ids = sorted(set(record.metal_ids) | metals)
    record.nucleotide_ids = sorted(set(record.nucleotide_ids) | nucleotides)
    record.partner_chain_ids = sorted(set(record.partner_chain_ids) | protein_partners)
    record.nucleic_acid_partner_chain_ids = sorted(
        set(record.nucleic_acid_partner_chain_ids) | nucleic_partners
    )
    protein_chain_count = 1 + len(protein_partners)
    record.oligomeric_state = _oligomeric_state(protein_chain_count)
    return ParsedChain(record=record, residues=residues)


class _ObservedResidue:
    """Internal residue representation before full-sequence expansion."""

    def __init__(
        self,
        *,
        residue_name: str,
        one_letter: str,
        auth_seq_id: int,
        label_seq_id: int | None,
        insertion_code: str,
        ca_coord: np.ndarray | None,
        occupancy: float | None,
        altloc: str | None,
        alternate_locations_present: bool,
        label_chain_id: str | None,
    ) -> None:
        self.residue_name = residue_name
        self.one_letter = one_letter
        self.auth_seq_id = auth_seq_id
        self.label_seq_id = label_seq_id
        self.insertion_code = insertion_code
        self.ca_coord = ca_coord
        self.occupancy = occupancy
        self.altloc = altloc
        self.alternate_locations_present = alternate_locations_present
        self.label_chain_id = label_chain_id


def _find_chain(model: gemmi.Model, auth_chain_id: str, label_chain_id: str | None) -> gemmi.Chain | None:
    """Resolve either author or label chain identifiers."""
    for chain in model:
        if chain.name == auth_chain_id:
            return chain
    if label_chain_id:
        for chain in model:
            if any(_subchain_name(residue) == label_chain_id for residue in chain):
                return chain
    return None


def _observed_protein_residues(chain: gemmi.Chain) -> list[_ObservedResidue]:
    """Extract protein residues and deterministically select each C-alpha altloc."""
    output: list[_ObservedResidue] = []
    for residue in chain:
        one_letter = residue_to_one_letter(residue.name)
        if one_letter is None:
            continue
        candidates = [atom for atom in residue if atom.name.strip() == "CA"]
        atom = _select_altloc(candidates)
        label_seq = _optional_int(getattr(residue, "label_seq", None))
        insertion = str(residue.seqid.icode).strip().replace("\x00", "")
        output.append(
            _ObservedResidue(
                residue_name=residue.name.upper(),
                one_letter=one_letter,
                auth_seq_id=int(residue.seqid.num),
                label_seq_id=label_seq,
                insertion_code=insertion,
                ca_coord=(
                    np.asarray([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float)
                    if atom is not None
                    else None
                ),
                occupancy=float(atom.occ) if atom is not None else None,
                altloc=_normalize_altloc(atom.altloc) if atom is not None else None,
                alternate_locations_present=(
                    len(candidates) > 1
                    or any(_normalize_altloc(candidate.altloc) for candidate in candidates)
                ),
                label_chain_id=_subchain_name(residue),
            )
        )
    return output


def _entity_sequence(structure: gemmi.Structure, observed: list[_ObservedResidue]) -> str:
    """Return the selected entity's full sequence, including unresolved positions."""
    subchains = {item.label_chain_id for item in observed if item.label_chain_id}
    for entity in structure.entities:
        entity_subchains = {str(value) for value in entity.subchains}
        if subchains & entity_subchains and entity.full_sequence:
            sequence = gemmi.one_letter_code(entity.full_sequence).replace(" ", "").upper()
            if sequence:
                return sequence
    return ""


def _merge_full_and_observed(
    full_sequence: str,
    observed: list[_ObservedResidue],
) -> list[ParsedResidue]:
    """Place observed residues at label-sequence positions when available."""
    by_label = {item.label_seq_id: item for item in observed if item.label_seq_id is not None}
    use_labels = bool(by_label) and max(by_label) <= len(full_sequence)
    residues: list[ParsedResidue] = []
    if use_labels:
        for index, one_letter in enumerate(full_sequence, start=1):
            item = by_label.get(index)
            residues.append(
                ParsedResidue(
                    index=index - 1,
                    residue_name=item.residue_name if item else one_letter,
                    one_letter=one_letter,
                    auth_seq_id=item.auth_seq_id if item else None,
                    label_seq_id=index,
                    insertion_code=item.insertion_code if item else "",
                    ca_coord=item.ca_coord if item else None,
                    occupancy=item.occupancy if item else None,
                    altloc=item.altloc if item else None,
                    alternate_locations_present=item.alternate_locations_present if item else False,
                )
            )
        return residues

    for index, item in enumerate(observed):
        residues.append(
            ParsedResidue(
                index=index,
                residue_name=item.residue_name,
                one_letter=item.one_letter,
                auth_seq_id=item.auth_seq_id,
                label_seq_id=item.label_seq_id,
                insertion_code=item.insertion_code,
                ca_coord=item.ca_coord,
                occupancy=item.occupancy,
                altloc=item.altloc,
                alternate_locations_present=item.alternate_locations_present,
            )
        )
    return residues


def _select_altloc(atoms: Iterable[gemmi.Atom]) -> gemmi.Atom | None:
    """Select highest occupancy, preferring blank then A when tied."""
    candidates = list(atoms)
    if not candidates:
        return None

    def key(atom: gemmi.Atom) -> tuple[float, int, str]:
        altloc = _normalize_altloc(atom.altloc)
        preference = 2 if altloc == "" else 1 if altloc == "A" else 0
        return (float(atom.occ), preference, altloc)

    return max(candidates, key=key)


def _selected_chain_environment(chain: gemmi.Chain) -> tuple[set[str], set[str], set[str]]:
    """Collect non-protein chemical IDs while retaining the raw structure separately."""
    ligands: set[str] = set()
    metals: set[str] = set()
    nucleotides: set[str] = set()
    for residue in chain:
        name = residue.name.strip().upper()
        if residue_to_one_letter(name) is not None or name in WATER_NAMES:
            continue
        if name in METAL_NAMES:
            metals.add(name)
        elif name in NUCLEOTIDE_NAMES:
            nucleotides.add(name)
        else:
            ligands.add(name)
    return ligands, metals, nucleotides


def _partner_chains(model: gemmi.Model, selected_chain_name: str) -> tuple[set[str], set[str]]:
    """Classify other polymer chains co-deposited with the endpoint chain."""
    proteins: set[str] = set()
    nucleic: set[str] = set()
    for chain in model:
        if chain.name == selected_chain_name:
            continue
        names = {residue.name.strip().upper() for residue in chain}
        if any(residue_to_one_letter(name) is not None for name in names):
            proteins.add(chain.name)
        elif names & NUCLEIC_NAMES:
            nucleic.add(chain.name)
    return proteins, nucleic


def _subchain_name(residue: gemmi.Residue) -> str | None:
    value = str(getattr(residue, "subchain", "") or "").strip()
    return value or None


def _optional_int(value: object) -> int | None:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _normalize_altloc(value: object) -> str:
    return str(value or "").strip().replace("\x00", "")


def _oligomeric_state(protein_chain_count: int) -> str:
    names = {1: "monomer", 2: "dimer", 3: "trimer", 4: "tetramer"}
    return names.get(protein_chain_count, f"{protein_chain_count}-mer")
