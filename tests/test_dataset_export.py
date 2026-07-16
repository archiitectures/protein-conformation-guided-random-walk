"""Standardized export and explicit-manifest pipeline tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from protein_conformer_dataset.config import DatasetConfig, QCConfig
from protein_conformer_dataset.export import export_endpoint_pair
from protein_conformer_dataset.models import StructureRecord
from protein_conformer_dataset.pair_selection import compare_pair
from protein_conformer_dataset.parsing import parse_mmcif_chain
from protein_conformer_dataset.pipeline import DatasetPipeline
from protein_conformer_dataset.sequence import choose_bioemu_sequence


def test_export_writes_standardized_endpoint_files(tmp_path: Path, mmcif_factory) -> None:
    path_a = mmcif_factory("export_a.cif", length=45)
    path_b = mmcif_factory("export_b.cif", length=45, deformation=2.0)
    record_a = StructureRecord("AAAA", "1", "A", "A", uniprot_id="P1", organism="X", file_path=str(path_a))
    record_b = StructureRecord("BBBB", "1", "A", "A", uniprot_id="P1", organism="X", file_path=str(path_b))
    chain_a = parse_mmcif_chain(path_a, record_a)
    chain_b = parse_mmcif_chain(path_b, record_b)
    pair = compare_pair("toy", chain_a, chain_b, QCConfig())
    sequence = choose_bioemu_sequence(chain_a.sequence, chain_b.sequence, "A" * 45)

    manifest_path = export_endpoint_pair(pair, chain_a, chain_b, sequence, tmp_path / "entry")

    assert manifest_path.exists()
    assert (tmp_path / "entry/processed/endpoint_A_ca.npy").exists()
    assert (tmp_path / "entry/processed/common_residue_mapping.csv").exists()
    assert (tmp_path / "entry/sequence/bioemu_selected.fasta").exists()
    assert np.load(tmp_path / "entry/processed/endpoint_A_aligned_ca.npy").shape == (45, 3)
    processed = parse_mmcif_chain(
        tmp_path / "entry/processed/endpoint_A_protein_only.cif",
        StructureRecord("AAAA", "1", "A", "A"),
    )
    assert processed.ca_count == 45
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["scientific_caveat"]
    assert manifest["paths"]["bioemu_sequence"] == "sequence/bioemu_selected.fasta"


def test_explicit_manifest_exports_every_row_and_resumes(
    tmp_path: Path, mmcif_factory
) -> None:
    source_a = mmcif_factory("manifest_a.cif", length=12)
    source_b = mmcif_factory("manifest_b.cif", length=12, deformation=2.0)
    manifest = tmp_path / "endpoints.csv"
    pd.DataFrame(
        [
            {
                "protein_id": "001",
                "protein_name": "toy",
                "uniprot_id": "",
                "pdb_a": "AAAA",
                "chain_a": "A",
                "pdb_b": "BBBB",
                "chain_b": "A",
                "state_a": "open",
                "state_b": "closed",
            }
        ]
    ).to_csv(manifest, index=False)
    config = DatasetConfig(
        output_dir=tmp_path / "dataset",
        cache_dir=tmp_path / "cache",
        resume=True,
        qc=QCConfig(minimum_common_ca_count=5, duplicate_rmsd_angstrom=0.1),
    )
    http = _FakeHttp({"AAAA": source_a, "BBBB": source_b})
    rcsb = _FakeRCSB()

    first = DatasetPipeline(config, http_client=http, rcsb_client=rcsb).run_manifest(manifest)
    calls_after_first = (http.download_calls, rcsb.entity_calls)
    second = DatasetPipeline(config, http_client=http, rcsb_client=rcsb).run_manifest(manifest)

    assert first.detailed_pairs_exported == 1
    assert second.detailed_pairs_exported == 1
    assert (http.download_calls, rcsb.entity_calls) == calls_after_first
    assert (tmp_path / "dataset/001/pairs/AAAA_A__BBBB_A/manifest.json").exists()
    dataset_manifest = pd.read_csv(tmp_path / "dataset/dataset_manifest.csv", dtype=str)
    assert dataset_manifest.loc[0, "protein_id"] == "001"


class _FakeHttp:
    def __init__(self, files: dict[str, Path]) -> None:
        self.files = files
        self.download_calls = 0

    def download_cif(self, pdb_id: str, *, force: bool = False) -> Path:
        self.download_calls += 1
        return self.files[pdb_id]


class _FakeRCSB:
    def __init__(self) -> None:
        self.entity_calls = 0

    def fetch_entry(self, pdb_id: str, *, force: bool = False):
        return {"rcsb_entry_container_identifiers": {"polymer_entity_ids": ["1"]}}

    def fetch_entity(
        self,
        pdb_id: str,
        entity_id: str,
        *,
        preferred_uniprot: str | None = None,
        preferred_auth_chain: str | None = None,
        force: bool = False,
    ) -> StructureRecord:
        self.entity_calls += 1
        return StructureRecord(
            pdb_id,
            entity_id,
            preferred_auth_chain or "A",
            "A",
            uniprot_id=preferred_uniprot,
            protein_name="toy",
            organism="Synthetic",
            experimental_method="X-RAY DIFFRACTION",
            resolution=2.0,
        )
