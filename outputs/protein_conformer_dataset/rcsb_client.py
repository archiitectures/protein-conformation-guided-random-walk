"""RCSB Search/Data API client for experimental polymer entities."""

from __future__ import annotations

import logging
from typing import Any

from protein_conformer_dataset.download import CachedHttpClient, DownloadError
from protein_conformer_dataset.models import StructureRecord


class RCSBClient:
    """Discover exact UniProt-mapped protein entities and retrieve metadata."""

    SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
    DATA_URL = "https://data.rcsb.org/rest/v1/core"

    def __init__(self, http: CachedHttpClient, logger: logging.Logger | None = None) -> None:
        self.http = http
        self.logger = logger or logging.getLogger("protein_conformer_dataset")

    def find_by_uniprot(self, accession: str, *, force: bool = False) -> list[StructureRecord]:
        """Return protein polymer entities exactly mapped to a UniProt accession."""
        normalized = accession.strip().upper()
        query = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "operator": "exact_match",
                            "value": normalized,
                            "attribute": (
                                "rcsb_polymer_entity_container_identifiers."
                                "reference_sequence_identifiers.database_accession"
                            ),
                        },
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "operator": "exact_match",
                            "value": "UniProt",
                            "attribute": (
                                "rcsb_polymer_entity_container_identifiers."
                                "reference_sequence_identifiers.database_name"
                            ),
                        },
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "operator": "exact_match",
                            "value": "Protein",
                            "attribute": "entity_poly.rcsb_entity_polymer_type",
                        },
                    },
                ],
            },
            "return_type": "polymer_entity",
            "request_options": {"paginate": {"start": 0, "rows": 10000}},
        }
        payload = self.http.request_json(
            "POST",
            self.SEARCH_URL,
            f"api/rcsb/search_uniprot_{normalized}.json",
            json_body=query,
            force=force,
        )
        identifiers = [item["identifier"] for item in payload.get("result_set", [])]
        records: list[StructureRecord] = []
        for identifier in identifiers:
            try:
                pdb_id, entity_id = identifier.split("_", maxsplit=1)
                record = self.fetch_entity(pdb_id, entity_id, preferred_uniprot=normalized, force=force)
                if record.uniprot_id == normalized:
                    records.append(record)
            except (DownloadError, KeyError, ValueError) as exc:
                self.logger.warning("Skipping RCSB entity %s: %s", identifier, exc)
        return _deduplicate_records(records)

    def records_for_pdb(self, pdb_id: str, *, force: bool = False) -> list[StructureRecord]:
        """Return exact UniProt-mapped protein entities from one PDB entry."""
        normalized = pdb_id.upper()
        entry = self.fetch_entry(normalized, force=force)
        entity_ids = entry.get("rcsb_entry_container_identifiers", {}).get("polymer_entity_ids", [])
        records: list[StructureRecord] = []
        for entity_id in entity_ids:
            try:
                record = self.fetch_entity(normalized, str(entity_id), force=force)
            except (DownloadError, KeyError, ValueError) as exc:
                self.logger.warning("Skipping RCSB entity %s_%s: %s", normalized, entity_id, exc)
                continue
            if record.uniprot_id:
                records.append(record)
        return _deduplicate_records(records)

    def fetch_entry(self, pdb_id: str, *, force: bool = False) -> dict[str, Any]:
        """Fetch entry-level RCSB metadata."""
        normalized = pdb_id.upper()
        return self.http.request_json(
            "GET",
            f"{self.DATA_URL}/entry/{normalized}",
            f"api/rcsb/entry_{normalized}.json",
            force=force,
        )

    def fetch_entity(
        self,
        pdb_id: str,
        entity_id: str,
        *,
        preferred_uniprot: str | None = None,
        preferred_auth_chain: str | None = None,
        force: bool = False,
    ) -> StructureRecord:
        """Fetch and normalize one polymer entity and one representative chain."""
        normalized = pdb_id.upper()
        entity = self.http.request_json(
            "GET",
            f"{self.DATA_URL}/polymer_entity/{normalized}/{entity_id}",
            f"api/rcsb/entity_{normalized}_{entity_id}.json",
            force=force,
        )
        entry = self.fetch_entry(normalized, force=force)
        identifiers = entity.get("rcsb_polymer_entity_container_identifiers", {})
        references = identifiers.get("reference_sequence_identifiers") or []
        uniprot_ids = [
            str(item.get("database_accession", "")).upper()
            for item in references
            if str(item.get("database_name", "")).lower() == "uniprot"
        ]
        chosen_uniprot = None
        if preferred_uniprot and preferred_uniprot.upper() in uniprot_ids:
            chosen_uniprot = preferred_uniprot.upper()
        elif len(uniprot_ids) == 1:
            chosen_uniprot = uniprot_ids[0]
        elif uniprot_ids:
            chosen_uniprot = sorted(uniprot_ids)[0]

        auth_ids = [str(value) for value in identifiers.get("auth_asym_ids") or []]
        label_ids = [str(value) for value in identifiers.get("asym_ids") or []]
        if not auth_ids and not label_ids:
            raise ValueError(f"RCSB entity {normalized}_{entity_id} has no chain identifiers")
        if preferred_auth_chain and preferred_auth_chain not in auth_ids:
            raise ValueError(
                f"Chain {preferred_auth_chain!r} does not belong to RCSB entity {normalized}_{entity_id}"
            )
        if preferred_auth_chain:
            auth_chain = preferred_auth_chain
            index = auth_ids.index(preferred_auth_chain)
        else:
            auth_chain = sorted(auth_ids)[0] if auth_ids else sorted(label_ids)[0]
            index = auth_ids.index(auth_chain) if auth_chain in auth_ids else 0
        label_chain = label_ids[index] if index < len(label_ids) else (sorted(label_ids)[0] if label_ids else None)
        sequence = str(entity.get("entity_poly", {}).get("pdbx_seq_one_letter_code_can", ""))
        sequence = "".join(sequence.split()).upper()
        organisms = entity.get("rcsb_entity_source_organism") or []
        descriptions = entity.get("rcsb_polymer_entity", {})
        entry_info = entry.get("rcsb_entry_info", {})
        resolutions = entry_info.get("resolution_combined") or []
        experiments = entry.get("exptl") or []
        accession = entry.get("rcsb_accession_info", {})
        mutations = _mutation_annotations(entity)
        return StructureRecord(
            pdb_id=normalized,
            entity_id=str(entity_id),
            auth_chain_id=auth_chain,
            label_chain_id=label_chain,
            uniprot_id=chosen_uniprot,
            protein_name=descriptions.get("pdbx_description"),
            organism=organisms[0].get("ncbi_scientific_name") if organisms else None,
            experimental_method=experiments[0].get("method") if experiments else None,
            resolution=float(resolutions[0]) if resolutions else None,
            deposition_date=accession.get("deposit_date") or accession.get("initial_release_date"),
            sequence=sequence,
            sequence_length=len(sequence),
            mutation_annotations=mutations,
            biological_assembly_id="1" if entry_info.get("assembly_count", 0) else None,
            metadata={"all_uniprot_ids": uniprot_ids, "rcsb_entity_id": f"{normalized}_{entity_id}"},
        )


def _mutation_annotations(entity: dict[str, Any]) -> list[str]:
    """Extract mutation-like RCSB feature descriptions when available."""
    annotations: list[str] = []
    for feature in entity.get("rcsb_polymer_entity_feature") or []:
        feature_type = str(feature.get("type", "")).lower()
        if "mutation" not in feature_type:
            continue
        description = feature.get("description") or feature.get("name") or feature_type
        annotations.append(str(description))
    mutation_count = entity.get("rcsb_polymer_entity", {}).get("rcsb_mutation_count")
    if mutation_count and not annotations:
        annotations.append(f"RCSB reports {mutation_count} engineered mutation(s)")
    return sorted(set(annotations))


def _deduplicate_records(records: list[StructureRecord]) -> list[StructureRecord]:
    """Remove repeated entity/crystallographic-copy records deterministically."""
    by_key: dict[tuple[str, str | None], StructureRecord] = {}
    for record in records:
        by_key.setdefault((record.pdb_id, record.entity_id), record)
    return sorted(by_key.values(), key=lambda item: (item.pdb_id, item.entity_id or ""))
