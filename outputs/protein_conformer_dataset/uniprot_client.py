"""Small UniProt REST client for canonical sequence metadata."""

from __future__ import annotations

from dataclasses import dataclass

from protein_conformer_dataset.download import CachedHttpClient


@dataclass(frozen=True, slots=True)
class UniProtRecord:
    """Canonical UniProt sequence and identity metadata."""

    accession: str
    sequence: str
    protein_name: str | None
    organism: str | None


class UniProtClient:
    """Retrieve exact UniProtKB records with local response caching."""

    BASE_URL = "https://rest.uniprot.org/uniprotkb"

    def __init__(self, http: CachedHttpClient) -> None:
        self.http = http

    def fetch(self, accession: str, *, force: bool = False) -> UniProtRecord:
        """Fetch one canonical UniProtKB entry."""
        normalized = accession.strip().upper()
        payload = self.http.request_json(
            "GET",
            f"{self.BASE_URL}/{normalized}.json",
            f"api/uniprot/{normalized}.json",
            force=force,
        )
        sequence = str(payload.get("sequence", {}).get("value", "")).replace(" ", "").upper()
        if not sequence:
            raise ValueError(f"UniProt entry {normalized} has no canonical sequence")
        description = payload.get("proteinDescription", {})
        recommended = description.get("recommendedName", {}).get("fullName", {}).get("value")
        submitted = description.get("submissionNames") or []
        protein_name = recommended
        if not protein_name and submitted:
            protein_name = submitted[0].get("fullName", {}).get("value")
        return UniProtRecord(
            accession=normalized,
            sequence=sequence,
            protein_name=protein_name,
            organism=payload.get("organism", {}).get("scientificName"),
        )
