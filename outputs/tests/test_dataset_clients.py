"""Offline cache and download validation tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from protein_conformer_dataset.download import CachedHttpClient, DownloadError


def test_json_response_is_cached_without_second_request(tmp_path: Path) -> None:
    session = _FakeSession(json_payload={"value": 7})
    client = CachedHttpClient(tmp_path, rate_seconds=0, session=session)

    first = client.request_json("GET", "https://example.test/data", "api/test.json")
    second = client.request_json("GET", "https://example.test/data", "api/test.json")

    assert first == second == {"value": 7}
    assert session.request_count == 1
    assert json.loads((tmp_path / "api/test.json").read_text(encoding="utf-8")) == {"value": 7}


def test_download_rejects_empty_or_corrupt_mmcif(tmp_path: Path) -> None:
    session = _FakeSession(content=b"not cif")
    client = CachedHttpClient(tmp_path, rate_seconds=0, session=session)

    with pytest.raises(DownloadError, match="empty or malformed"):
        client.download_cif("1ABC")


class _FakeResponse:
    def __init__(self, *, json_payload=None, content: bytes = b"") -> None:
        self._json_payload = json_payload
        self.content = content

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._json_payload


class _FakeSession(requests.Session):
    def __init__(self, *, json_payload=None, content: bytes = b"") -> None:
        super().__init__()
        self.json_payload = json_payload
        self.content = content
        self.request_count = 0

    def request(self, method, url, **kwargs):
        self.request_count += 1
        return _FakeResponse(json_payload=self.json_payload)

    def get(self, url, **kwargs):
        self.request_count += 1
        return _FakeResponse(content=self.content)
