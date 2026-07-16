"""End-to-end orchestration for conformational endpoint dataset construction."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from protein_conformer_dataset.alignment import map_to_canonical
from protein_conformer_dataset.checkpoint import StateStore
from protein_conformer_dataset.config import DatasetConfig
from protein_conformer_dataset.download import CachedHttpClient, DownloadError
from protein_conformer_dataset.export import (
    export_endpoint_pair,
    pair_table_row,
    structure_table_row,
    write_dataset_tables,
)
from protein_conformer_dataset.logging_utils import configure_logging
from protein_conformer_dataset.models import EndpointPair, PairStatus, ParsedChain, StructureRecord
from protein_conformer_dataset.pair_selection import (
    compare_all_pairs,
    compare_pair,
    prefilter_structure_records,
    rank_candidate_pairs,
)
from protein_conformer_dataset.parsing import StructureParseError, parse_mmcif_chain
from protein_conformer_dataset.rcsb_client import RCSBClient
from protein_conformer_dataset.sequence import SequenceChoice, choose_bioemu_sequence
from protein_conformer_dataset.uniprot_client import UniProtClient, UniProtRecord


@dataclass(slots=True)
class RunSummary:
    """Aggregate counts and recoverable failures from one dataset run."""

    identifiers_requested: int = 0
    proteins_resolved: int = 0
    structures_downloaded: int = 0
    structures_parsed: int = 0
    candidate_pairs_generated: int = 0
    accepted_pairs: int = 0
    manual_review_pairs: int = 0
    rejected_pairs: int = 0
    detailed_pairs_exported: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready summary."""
        return asdict(self)

    def format(self) -> str:
        """Return a compact command-line summary."""
        return "\n".join(
            [
                f"Identifiers requested: {self.identifiers_requested}",
                f"Proteins resolved: {self.proteins_resolved}",
                f"Structures downloaded: {self.structures_downloaded}",
                f"Structures parsed: {self.structures_parsed}",
                f"Candidate pairs generated: {self.candidate_pairs_generated}",
                f"Accepted endpoint pairs: {self.accepted_pairs}",
                f"Manual-review pairs: {self.manual_review_pairs}",
                f"Rejected pairs: {self.rejected_pairs}",
                f"Detailed pairs exported: {self.detailed_pairs_exported}",
                f"Recoverable errors: {len(self.errors)}",
            ]
        )


class DatasetPipeline:
    """Build candidate experimental endpoint datasets from supported identifiers."""

    def __init__(
        self,
        config: DatasetConfig,
        *,
        http_client: CachedHttpClient | None = None,
        rcsb_client: RCSBClient | None = None,
        uniprot_client: UniProtClient | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Create a pipeline with injectable clients for offline tests."""
        self.config = config
        self.logger = logger or configure_logging(config.output_dir)
        self.http = http_client or CachedHttpClient(
            config.cache_dir,
            timeout_seconds=config.request_timeout_seconds,
            rate_seconds=config.request_rate_seconds,
            retries=config.request_retries,
            logger=self.logger,
        )
        self.rcsb = rcsb_client or RCSBClient(self.http, self.logger)
        self.uniprot = uniprot_client or UniProtClient(self.http)
        self._pair_rows: list[dict[str, Any]] = []
        self._structure_rows: list[dict[str, Any]] = []
        self._seen_structures: set[str] = set()

    def run_uniprot_ids(self, accessions: Iterable[str]) -> RunSummary:
        """Discover, compare, and export endpoints for exact UniProt accessions."""
        identifiers = _unique_normalized(accessions)
        summary = RunSummary(identifiers_requested=len(identifiers))
        state = self._state_store("uniprot", identifiers)
        for accession in identifiers:
            started = time.monotonic()
            try:
                metadata = state.load(accession, "metadata", enabled=self._resume_enabled())
                if metadata is None:
                    canonical = self.uniprot.fetch(accession, force=self.config.force)
                    records = self.rcsb.find_by_uniprot(accession, force=self.config.force)
                    records = prefilter_structure_records(
                        records, len(canonical.sequence), self.config.max_structures_per_protein
                    )
                    metadata = {
                        "canonical": asdict(canonical),
                        "records": [record.to_dict() for record in records],
                    }
                    state.save(accession, "metadata", metadata)
                canonical = UniProtRecord(**metadata["canonical"])
                records = [StructureRecord.from_dict(item) for item in metadata["records"]]
                self._process_automatic_group(accession, canonical, records, state, summary)
            except Exception as exc:
                self._record_error(summary, accession, exc)
            finally:
                self.logger.info("Finished %s in %.2f seconds", accession, time.monotonic() - started)
        self._finish(summary)
        return summary

    def run_pdb_ids(self, pdb_ids: Iterable[str]) -> RunSummary:
        """Group explicit PDB entries by exact UniProt mapping and discover pairs."""
        identifiers = _unique_normalized(pdb_ids)
        summary = RunSummary(identifiers_requested=len(identifiers))
        state = self._state_store("pdb", identifiers)
        grouped: dict[str, list[StructureRecord]] = {}
        for pdb_id in identifiers:
            try:
                for record in self.rcsb.records_for_pdb(pdb_id, force=self.config.force):
                    if record.uniprot_id:
                        grouped.setdefault(record.uniprot_id, []).append(record)
            except Exception as exc:
                self._record_error(summary, pdb_id, exc)
        for accession, records in grouped.items():
            try:
                canonical = self.uniprot.fetch(accession, force=self.config.force)
                records = prefilter_structure_records(
                    records, len(canonical.sequence), self.config.max_structures_per_protein
                )
                self._process_automatic_group(accession, canonical, records, state, summary)
            except Exception as exc:
                self._record_error(summary, accession, exc)
        self._finish(summary)
        return summary

    def run_manifest(self, manifest_path: str | Path) -> RunSummary:
        """Process every explicit endpoint row, including rows that fail QC."""
        path = Path(manifest_path)
        dataframe = pd.read_csv(path, dtype=str, keep_default_na=False)
        required = {"protein_id", "pdb_a", "chain_a", "pdb_b", "chain_b"}
        missing = required - set(dataframe.columns)
        if missing:
            raise ValueError(f"Manifest is missing required columns: {', '.join(sorted(missing))}")
        rows = dataframe.to_dict(orient="records")
        summary = RunSummary(identifiers_requested=len(rows))
        state = self._state_store("manifest", rows)
        for row_index, row in enumerate(rows):
            unit_id = f"row_{row_index:05d}_{row['protein_id']}"
            try:
                self._process_manifest_row(unit_id, row, state, summary)
            except Exception as exc:
                self._record_error(summary, unit_id, exc)
        self._finish(summary)
        return summary

    def _process_automatic_group(
        self,
        protein_id: str,
        canonical: UniProtRecord,
        records: list[StructureRecord],
        state: StateStore,
        summary: RunSummary,
    ) -> None:
        """Process one automatically discovered structure group."""
        chains = self._download_and_parse(protein_id, records, canonical.sequence, state, summary)
        if len(chains) < 2:
            raise ValueError(f"Fewer than two usable structures were found for {protein_id}")
        summary.proteins_resolved += 1
        comparisons_payload = state.load(protein_id, "comparisons", enabled=self._resume_enabled())
        if comparisons_payload is None:
            pairs = compare_all_pairs(protein_id, chains, self.config.qc)
            state.save(protein_id, "comparisons", [pair.to_dict() for pair in pairs])
        else:
            pairs = [EndpointPair.from_dict(item) for item in comparisons_payload]
        ranked = rank_candidate_pairs(
            pairs, self.config.ranking, self.config.qc.duplicate_rmsd_angstrom
        )
        for rank, pair in enumerate(ranked[: self.config.top_k_pairs], start=1):
            row = pair_table_row(pair)
            row["candidate_rank"] = rank
            self._pair_rows.append(row)
        ranked_ids = {pair.pair_id for pair in ranked[: self.config.top_k_pairs]}
        for pair in pairs:
            if pair.pair_id not in ranked_ids:
                self._pair_rows.append(pair_table_row(pair))
        if ranked:
            selected = ranked[0]
            chain_a = _chain_for_record(chains, selected.structure_a)
            chain_b = _chain_for_record(chains, selected.structure_b)
            sequence_choice = choose_bioemu_sequence(
                chain_a.sequence,
                chain_b.sequence,
                canonical.sequence,
                minimum_identity=self.config.qc.minimum_sequence_identity,
                minimum_coverage=self.config.qc.minimum_aligned_coverage,
            )
            destination = self.config.output_dir / _safe_id(protein_id)
            export_payload = state.load(protein_id, "export", enabled=self._resume_enabled())
            manifest_path = destination / "manifest.json"
            if export_payload is None or not manifest_path.exists():
                export_endpoint_pair(selected, chain_a, chain_b, sequence_choice, destination)
                state.save(protein_id, "export", {"manifest_path": str(manifest_path)})
            self._attach_output_path(selected.pair_id, manifest_path)
            summary.detailed_pairs_exported += 1
        self._accumulate_counts(summary, pairs)

    def _process_manifest_row(
        self,
        unit_id: str,
        row: dict[str, str],
        state: StateStore,
        summary: RunSummary,
    ) -> None:
        """Resolve and export one user-supplied endpoint pair."""
        metadata = state.load(unit_id, "metadata", enabled=self._resume_enabled())
        if metadata is None:
            uniprot_id = row.get("uniprot_id") or None
            canonical: UniProtRecord | None = None
            if uniprot_id:
                canonical = self.uniprot.fetch(uniprot_id, force=self.config.force)
            record_a = self._resolve_manifest_record(row["pdb_a"], row["chain_a"], uniprot_id)
            record_b = self._resolve_manifest_record(row["pdb_b"], row["chain_b"], uniprot_id)
            if row.get("protein_name"):
                record_a.protein_name = row["protein_name"]
                record_b.protein_name = row["protein_name"]
            metadata = {
                "canonical": asdict(canonical) if canonical else None,
                "records": [record_a.to_dict(), record_b.to_dict()],
            }
            state.save(unit_id, "metadata", metadata)
        canonical = UniProtRecord(**metadata["canonical"]) if metadata["canonical"] else None
        record_a, record_b = [StructureRecord.from_dict(item) for item in metadata["records"]]
        records = [record_a, record_b]
        chains = self._download_and_parse(
            unit_id, records, canonical.sequence if canonical else None, state, summary
        )
        if len(chains) != 2:
            raise ValueError("One or both explicit endpoint structures could not be parsed")
        comparisons_payload = state.load(unit_id, "comparisons", enabled=self._resume_enabled())
        if comparisons_payload is None:
            pair = compare_pair(
                row["protein_id"],
                chains[0],
                chains[1],
                self.config.qc,
                state_a=row.get("state_a") or None,
                state_b=row.get("state_b") or None,
            )
            pair.candidate_score = 0.0
            state.save(unit_id, "comparisons", [pair.to_dict()])
        else:
            pair = EndpointPair.from_dict(comparisons_payload[0])
        sequence_choice = choose_bioemu_sequence(
            chains[0].sequence,
            chains[1].sequence,
            canonical.sequence if canonical else None,
            minimum_identity=self.config.qc.minimum_sequence_identity,
            minimum_coverage=self.config.qc.minimum_aligned_coverage,
        )
        destination = (
            self.config.output_dir / _safe_id(row["protein_id"]) / "pairs" / _safe_id(pair.pair_id)
        )
        manifest_path = destination / "manifest.json"
        export_payload = state.load(unit_id, "export", enabled=self._resume_enabled())
        if export_payload is None or not manifest_path.exists():
            export_endpoint_pair(pair, chains[0], chains[1], sequence_choice, destination)
            state.save(unit_id, "export", {"manifest_path": str(manifest_path)})
        pair_row = pair_table_row(pair, manifest_path)
        pair_row["candidate_rank"] = 1
        self._pair_rows.append(pair_row)
        summary.proteins_resolved += 1
        summary.detailed_pairs_exported += 1
        self._accumulate_counts(summary, [pair])

    def _download_and_parse(
        self,
        unit_id: str,
        records: list[StructureRecord],
        canonical_sequence: str | None,
        state: StateStore,
        summary: RunSummary,
    ) -> list[ParsedChain]:
        """Download and parse structures with independent failures and checkpoints."""
        download_payload = state.load(unit_id, "downloads", enabled=self._resume_enabled())
        if download_payload is None:
            downloaded: list[StructureRecord] = []
            for record in records:
                try:
                    record.file_path = str(self.http.download_cif(record.pdb_id, force=self.config.force))
                    downloaded.append(record)
                except DownloadError as exc:
                    self.logger.warning("Skipping coordinate download %s: %s", record.structure_id, exc)
            state.save(unit_id, "downloads", [record.to_dict() for record in downloaded])
        else:
            downloaded = [StructureRecord.from_dict(item) for item in download_payload]
        summary.structures_downloaded += len(downloaded)

        parsed_payload = state.load(unit_id, "parsing", enabled=self._resume_enabled())
        if parsed_payload is None:
            chains: list[ParsedChain] = []
            for record in downloaded:
                try:
                    chain = parse_mmcif_chain(record.file_path or "", record)
                    if canonical_sequence:
                        map_to_canonical(chain, canonical_sequence)
                    chains.append(chain)
                except (StructureParseError, ValueError) as exc:
                    self.logger.warning("Skipping parsed structure %s: %s", record.structure_id, exc)
            state.save(unit_id, "parsing", [chain.to_dict() for chain in chains])
        else:
            chains = [ParsedChain.from_dict(item) for item in parsed_payload]
        summary.structures_parsed += len(chains)
        for chain in chains:
            if chain.record.structure_id not in self._seen_structures:
                self._structure_rows.append(structure_table_row(chain.record))
                self._seen_structures.add(chain.record.structure_id)
        return chains

    def _resolve_manifest_record(
        self, pdb_id: str, auth_chain_id: str, uniprot_id: str | None
    ) -> StructureRecord:
        """Resolve an author chain to its RCSB polymer entity."""
        entry = self.rcsb.fetch_entry(pdb_id, force=self.config.force)
        entity_ids = entry.get("rcsb_entry_container_identifiers", {}).get("polymer_entity_ids", [])
        for entity_id in entity_ids:
            try:
                return self.rcsb.fetch_entity(
                    pdb_id,
                    str(entity_id),
                    preferred_uniprot=uniprot_id,
                    preferred_auth_chain=auth_chain_id,
                    force=self.config.force,
                )
            except ValueError:
                continue
        raise ValueError(f"Could not resolve PDB {pdb_id} author chain {auth_chain_id}")

    def _state_store(self, mode: str, inputs: Any) -> StateStore:
        config = self.config.to_dict()
        config.pop("resume", None)
        config.pop("force", None)
        return StateStore(self.config.output_dir, {"mode": mode, "inputs": inputs, "config": config})

    def _resume_enabled(self) -> bool:
        return self.config.resume and not self.config.force

    def _attach_output_path(self, pair_id: str, path: Path) -> None:
        for row in reversed(self._pair_rows):
            row_pair_id = f"{row['pdb_a']}_{row['chain_a']}__{row['pdb_b']}_{row['chain_b']}"
            if row_pair_id == pair_id:
                row["output_path"] = str(path)
                return

    def _accumulate_counts(self, summary: RunSummary, pairs: list[EndpointPair]) -> None:
        summary.candidate_pairs_generated += len(pairs)
        summary.accepted_pairs += sum(pair.quality.status is PairStatus.ACCEPTED for pair in pairs)
        summary.manual_review_pairs += sum(
            pair.quality.status is PairStatus.MANUAL_REVIEW for pair in pairs
        )
        summary.rejected_pairs += sum(pair.quality.status is PairStatus.REJECTED for pair in pairs)

    def _record_error(self, summary: RunSummary, identifier: str, exc: Exception) -> None:
        message = f"{identifier}: {exc}"
        summary.errors.append(message)
        self.logger.error(message)

    def _finish(self, summary: RunSummary) -> None:
        write_dataset_tables(
            self.config.output_dir,
            self._pair_rows,
            self._structure_rows,
            self.config.to_dict(),
        )
        summary_path = self.config.output_dir / "run_summary.json"
        summary_path.write_text(json.dumps(summary.to_dict(), indent=2) + "\n", encoding="utf-8")
        self.logger.info("\n%s", summary.format())


def _chain_for_record(chains: list[ParsedChain], record: StructureRecord) -> ParsedChain:
    for chain in chains:
        if chain.record.structure_id == record.structure_id:
            return chain
    raise KeyError(f"Parsed chain not found for {record.structure_id}")


def _unique_normalized(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value.strip().upper() for value in values if value.strip()))


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()) or "unnamed"
