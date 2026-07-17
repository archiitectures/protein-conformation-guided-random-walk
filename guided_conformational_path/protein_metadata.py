"""Resolve safe protein labels for visualization titles and HTML headings."""

from __future__ import annotations

import html
import json
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

_CACHE_DIR = Path(tempfile.gettempdir()) / "guided_conformational_path_cache"
_UNIPROT_CACHE_PATH = _CACHE_DIR / "uniprot_metadata_cache.json"
_UNIPROT_URL_TEMPLATE = "https://www.uniprot.org/uniprotkb/{accession}/entry"


def is_valid_uniprot_accession(accession: str | None) -> bool:
    """Return whether the string matches a conservative UniProt accession shape."""
    if not accession:
        return False
    value = accession.strip().upper()
    if not value:
        return False
    if len(value) == 6:
        return (
            (
                value[0] in "OPQ"
                and value[1].isdigit()
                and value[2:5].isalnum()
                and value[5].isdigit()
            )
            or (
                value[0].isalnum()
                and value[0] not in "OQ"
                and value[0].isalpha()
                and value[1].isdigit()
                and value[2].isalnum()
                and value[3:5].isalnum()
                and value[5].isdigit()
            )
        )
    if len(value) == 10:
        return (
            value[0].isalpha()
            and value[1].isdigit()
            and value[2].isalnum()
            and value[3:5].isalnum()
            and value[5].isdigit()
            and value[6].isalnum()
            and value[7:9].isalnum()
            and value[9].isdigit()
        )
    return False


def resolve_protein_metadata(
    summary: dict[str, Any],
    *,
    timeout_seconds: float = 2.0,
) -> dict[str, Any]:
    """Resolve display metadata once, preferring local summary fields and cached API data."""
    existing = summary.get("protein_metadata")
    if isinstance(existing, dict) and existing.get("display_label"):
        return existing

    input_metadata = summary.get("input") if isinstance(summary.get("input"), dict) else {}
    accession = _extract_accession(input_metadata)
    name = _extract_name(input_metadata)
    source = "local"

    if not name and accession:
        cached = _read_uniprot_cache().get(accession)
        if isinstance(cached, str) and cached.strip():
            name = cached.strip()
            source = "cache"
        else:
            repository_name = _read_repository_uniprot_name(accession)
            if repository_name:
                name = repository_name
                source = "repository_cache"
            else:
                fetched_name = _fetch_uniprot_name(accession, timeout_seconds=timeout_seconds)
                if fetched_name:
                    name = fetched_name
                    source = "uniprot_api"
                    _write_uniprot_cache_entry(accession, fetched_name)
                else:
                    source = "accession_only"

    metadata = {
        "name": name,
        "accession": accession,
        "display_label": _display_label(name, accession),
        "uniprot_url": _uniprot_url(accession),
        "source": source,
    }
    summary["protein_metadata"] = metadata
    return metadata


def protein_heading_html(summary: dict[str, Any]) -> str:
    """Return a safe HTML heading fragment for a resolved protein label."""
    metadata = resolve_protein_metadata(summary)
    label = html.escape(str(metadata.get("display_label") or ""))
    url = metadata.get("uniprot_url")
    if url:
        return f'<a href="{html.escape(str(url), quote=True)}">{label}</a>'
    return label


def protein_title_prefix(summary: dict[str, Any]) -> str:
    """Return a compact text prefix for animation titles."""
    metadata = resolve_protein_metadata(summary)
    label = str(metadata.get("display_label") or "").strip()
    return f"{label} | " if label else ""


def _extract_accession(input_metadata: dict[str, Any]) -> str | None:
    candidates = [
        input_metadata.get("uniprot_id"),
        input_metadata.get("protein_id"),
        _sequence_accession(input_metadata.get("bioemu_sequence")),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        value = str(candidate).strip().upper()
        if is_valid_uniprot_accession(value):
            return value
    return None


def _extract_name(input_metadata: dict[str, Any]) -> str | None:
    candidates = [
        input_metadata.get("protein_name"),
        input_metadata.get("name"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _sequence_accession(sequence_metadata: Any) -> str | None:
    if not isinstance(sequence_metadata, dict):
        return None
    sequence_id = sequence_metadata.get("id")
    if not isinstance(sequence_id, str) or not sequence_id.strip():
        return None
    accession = sequence_id.split("_", 1)[0].strip().upper()
    return accession if is_valid_uniprot_accession(accession) else None


def _display_label(name: str | None, accession: str | None) -> str:
    if name and accession:
        return f"{name} ({accession})"
    if name:
        return name
    if accession:
        return accession
    return ""


def _uniprot_url(accession: str | None) -> str | None:
    if not is_valid_uniprot_accession(accession):
        return None
    safe_accession = urllib.parse.quote(str(accession).upper(), safe="")
    return _UNIPROT_URL_TEMPLATE.format(accession=safe_accession)


def _read_uniprot_cache() -> dict[str, str]:
    try:
        return json.loads(_UNIPROT_CACHE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}


def _write_uniprot_cache_entry(accession: str, name: str) -> None:
    try:
        _UNIPROT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        cache = _read_uniprot_cache()
        cache[str(accession).upper()] = str(name)
        _UNIPROT_CACHE_PATH.write_text(
            json.dumps(cache, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except OSError:
        # Protein labels are optional visualization metadata; cache failures are non-fatal.
        return


def _repository_uniprot_cache_candidates(accession: str) -> list[Path]:
    """Return safe repository-relative UniProt payload candidates."""
    relative_path = Path("data") / "cache" / "api" / "uniprot" / f"{accession}.json"
    candidates = [Path.cwd() / relative_path, Path(__file__).resolve().parents[1] / relative_path]
    return list(dict.fromkeys(candidates))


def _read_repository_uniprot_name(accession: str) -> str | None:
    """Read a protein name from an existing dataset API cache payload."""
    for candidate in _repository_uniprot_cache_candidates(accession):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError):
            continue
        if isinstance(payload, dict):
            name = _extract_uniprot_name(payload)
            if name:
                return name
    return None


def _fetch_uniprot_name(accession: str, *, timeout_seconds: float) -> str | None:
    """Fetch one protein recommended name from UniProt; failures return None."""
    if not is_valid_uniprot_accession(accession):
        return None
    url = (
        "https://rest.uniprot.org/uniprotkb/"
        f"{urllib.parse.quote(accession, safe='')}?format=json"
    )
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "guided-conformational-path/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError):
        return None
    return _extract_uniprot_name(payload)


def _extract_uniprot_name(payload: dict[str, Any]) -> str | None:
    protein_description = payload.get("proteinDescription")
    if not isinstance(protein_description, dict):
        return None
    recommended = protein_description.get("recommendedName")
    if isinstance(recommended, dict):
        full_name = recommended.get("fullName")
        if isinstance(full_name, dict):
            value = full_name.get("value")
            if isinstance(value, str) and value.strip():
                return value.strip()
    submission = protein_description.get("submissionNames")
    if isinstance(submission, list):
        for item in submission:
            if not isinstance(item, dict):
                continue
            full_name = item.get("fullName")
            if isinstance(full_name, dict):
                value = full_name.get("value")
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return None
