"""Cache-aware HTTP and coordinate download utilities."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class DownloadError(RuntimeError):
    """Raised when a remote resource cannot be retrieved or validated."""


class CachedHttpClient:
    """Shared HTTP session with retries, rate limiting, and disk caching."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        timeout_seconds: float = 30.0,
        rate_seconds: float = 0.1,
        retries: int = 3,
        session: requests.Session | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Create a client; an injected session keeps tests independent of the network."""
        self.cache_dir = Path(cache_dir)
        self.timeout_seconds = timeout_seconds
        self.rate_seconds = rate_seconds
        self.logger = logger or logging.getLogger("protein_conformer_dataset")
        self.session = session or requests.Session()
        if session is None:
            retry = Retry(
                total=retries,
                connect=retries,
                read=retries,
                status=retries,
                backoff_factor=0.5,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=frozenset({"GET", "POST"}),
            )
            adapter = HTTPAdapter(max_retries=retry)
            self.session.mount("https://", adapter)
            self.session.mount("http://", adapter)
        self.session.headers.setdefault(
            "User-Agent",
            "protein-conformer-dataset/0.1 (research prototype; contact via project repository)",
        )

    def request_json(
        self,
        method: str,
        url: str,
        cache_path: str | Path,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        force: bool = False,
    ) -> Any:
        """Return cached JSON or issue and cache an HTTP request."""
        path = self.cache_dir / Path(cache_path)
        if path.exists() and path.stat().st_size > 0 and not force:
            self.logger.debug("API cache hit: %s", path)
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.logger.warning("Ignoring corrupt JSON cache file: %s", path)

        self.logger.info("API %s %s", method.upper(), url)
        self._wait()
        try:
            response = self.session.request(
                method.upper(),
                url,
                params=params,
                json=json_body,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise DownloadError(f"Failed to retrieve JSON from {url}: {exc}") from exc
        _write_json_atomic(path, payload)
        return payload

    def download_cif(self, pdb_id: str, *, force: bool = False) -> Path:
        """Download and validate an RCSB mmCIF coordinate file."""
        normalized = pdb_id.upper()
        destination = self.cache_dir / "structures" / f"{normalized}.cif"
        if destination.exists() and destination.stat().st_size > 64 and not force:
            self.logger.debug("Coordinate cache hit: %s", destination)
            return destination

        url = f"https://files.rcsb.org/download/{normalized}.cif"
        self.logger.info("Downloading %s", url)
        self._wait()
        try:
            response = self.session.get(url, timeout=self.timeout_seconds)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise DownloadError(f"Failed to download {normalized}: {exc}") from exc
        content = response.content
        if len(content) <= 64 or b"data_" not in content[:4096]:
            raise DownloadError(f"Downloaded mmCIF for {normalized} is empty or malformed")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".cif.tmp")
        temporary.write_bytes(content)
        temporary.replace(destination)
        return destination

    def _wait(self) -> None:
        """Apply a small polite delay between uncached calls."""
        if self.rate_seconds > 0:
            time.sleep(self.rate_seconds)


def _write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON through a neighboring temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
