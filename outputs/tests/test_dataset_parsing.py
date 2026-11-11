"""mmCIF parser tests for structural edge cases."""

from __future__ import annotations

from pathlib import Path

import pytest

from protein_conformer_dataset.models import StructureRecord
from protein_conformer_dataset.parsing import StructureParseError, parse_mmcif_chain


def test_parse_mmcif_preserves_missing_residues_and_insertion_codes(mmcif_factory) -> None:
    path = mmcif_factory("missing.cif", length=6, missing={3}, insertion_codes={4: "A"})
    chain = parse_mmcif_chain(path, StructureRecord("TEST", "1", "A", "A"))

    assert len(chain.residues) == 6
    assert chain.residues[2].ca_coord is None
    assert chain.residues[3].insertion_code == "A"
    assert chain.record.observed_residue_count == 5


def test_parse_mmcif_selects_highest_occupancy_altloc(mmcif_factory) -> None:
    path = mmcif_factory("altloc.cif", altloc_position=2)
    chain = parse_mmcif_chain(path, StructureRecord("TEST", "1", "A", "A"))

    assert chain.residues[1].altloc == "B"
    assert chain.residues[1].occupancy == pytest.approx(0.8)
    assert chain.residues[1].ca_coord[1] == pytest.approx(1.0)
    assert chain.record.alternate_locations_present is True


def test_parse_mmcif_records_models_ligands_and_partner_chains(mmcif_factory) -> None:
    path = mmcif_factory("environment.cif", models=2, ligand="ATP", partner_chain=True)
    chain = parse_mmcif_chain(path, StructureRecord("TEST", "1", "A", "A"))

    assert chain.record.model_count == 2
    assert chain.record.nucleotide_ids == ["ATP"]
    assert chain.record.partner_chain_ids == ["B"]
    assert chain.record.oligomeric_state == "dimer"


@pytest.mark.parametrize("content", ["", "not an mmcif file\n"])
def test_parse_mmcif_rejects_empty_or_corrupt_file(tmp_path: Path, content: str) -> None:
    path = tmp_path / "bad.cif"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(StructureParseError):
        parse_mmcif_chain(path, StructureRecord("BAD1", "1", "A"))
