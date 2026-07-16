"""Candidate endpoint ranking tests."""

from __future__ import annotations

from protein_conformer_dataset.config import QCConfig, RankingConfig
from protein_conformer_dataset.models import ParsedChain, StructureRecord
from protein_conformer_dataset.pair_selection import (
    compare_pair,
    prefilter_structure_records,
    rank_candidate_pairs,
)
from protein_conformer_dataset.parsing import parse_mmcif_chain


def test_prefilter_records_prefers_length_match_then_resolution() -> None:
    records = [
        StructureRecord("AAAA", "1", "A", sequence_length=80, resolution=1.0),
        StructureRecord("BBBB", "1", "A", sequence_length=100, resolution=3.0),
        StructureRecord("CCCC", "1", "A", sequence_length=100, resolution=1.5),
    ]

    selected = prefilter_structure_records(records, canonical_sequence_length=100, maximum=2)

    assert [record.pdb_id for record in selected] == ["CCCC", "BBBB"]


def test_ranking_excludes_near_duplicate_pair(mmcif_factory) -> None:
    path_a = mmcif_factory("rank_a.cif", length=45, deformation=0.0)
    path_b = mmcif_factory("rank_b.cif", length=45, deformation=0.1)
    path_c = mmcif_factory("rank_c.cif", length=45, deformation=3.0)
    chains = [
        parse_mmcif_chain(path, StructureRecord(pdb_id, "1", "A", "A", uniprot_id="P1", organism="X"))
        for path, pdb_id in ((path_a, "AAAA"), (path_b, "BBBB"), (path_c, "CCCC"))
    ]
    qc = QCConfig(duplicate_rmsd_angstrom=0.5)
    pairs = [
        compare_pair("P1", chains[0], chains[1], qc),
        compare_pair("P1", chains[0], chains[2], qc),
    ]

    ranked = rank_candidate_pairs(pairs, RankingConfig(), duplicate_rmsd_angstrom=0.5)

    assert len(ranked) == 1
    assert ranked[0].structure_b.pdb_id == "CCCC"
