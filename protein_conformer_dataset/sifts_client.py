"""Best-effort PDBe/SIFTS mapping client.

SIFTS enrichment is useful for canonical residue positions, but a temporary
mapping-service failure must not invalidate coordinate processing. Sequence
alignment remains the required correspondence mechanism.
"""

from __future__ import annotations

from typing import Any

from protein_conformer_dataset.download import CachedHttpClient


class SIFTSClient:
    """Retrieve cached PDB-to-UniProt residue mapping metadata from PDBe."""

    BASE_URL = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot"

    def __init__(self, http: CachedHttpClient) -> None:
        self.http = http

    def fetch(self, pdb_id: str, *, force: bool = False) -> dict[str, Any]:
        """Return the raw PDBe mapping response for a PDB entry."""
        normalized = pdb_id.lower()
        payload = self.http.request_json(
            "GET",
            f"{self.BASE_URL}/{normalized}",
            f"api/sifts/{normalized}.json",
            force=force,
        )
        return payload.get(normalized, payload)
