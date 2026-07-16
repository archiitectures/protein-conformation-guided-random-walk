from __future__ import annotations

import os
from pathlib import Path

import pytest

from parallel_bioemu_tools.hf_storage import HuggingFaceStorage, resolve_hf_token


class MockApi:
    def __init__(self) -> None:
        self.uploaded: list[tuple[str, str]] = []
        self.uploaded_files: list[tuple[str, str]] = []
        self.downloaded: list[str] = []
        self.files = {
            "run/protein/pair/n_16/manifest.json",
            "run/protein/pair/n_16/nodes/sample_000.pdb",
        }

    def upload_folder(
        self,
        *,
        repo_id: str,
        repo_type: str,
        folder_path: str,
        path_in_repo: str,
        ignore_patterns: list[str] | None = None,
    ) -> None:
        self.uploaded.append((folder_path, path_in_repo))
        root = Path(folder_path)
        self.files.update(
            f"{path_in_repo}/{path.relative_to(root).as_posix()}"
            for path in root.rglob("*")
            if path.is_file() and path.name not in set(ignore_patterns or [])
        )

    def upload_file(
        self,
        *,
        repo_id: str,
        repo_type: str,
        path_or_fileobj: str,
        path_in_repo: str,
    ) -> None:
        self.uploaded_files.append((path_or_fileobj, path_in_repo))
        self.files.add(path_in_repo)

    def hf_hub_download(self, *, repo_id: str, repo_type: str, filename: str, local_dir: str) -> str:
        self.downloaded.append(filename)
        path = os.path.join(local_dir, os.path.basename(filename))
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("{}\n")
        return path

    def list_repo_files(self, *, repo_id: str, repo_type: str) -> list[str]:
        return sorted(self.files)


def test_hf_token_comes_only_from_env(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    assert resolve_hf_token() == "secret-token"
    monkeypatch.delenv("HF_TOKEN")
    with pytest.raises(Exception):
        resolve_hf_token()


def test_hf_storage_uses_mocked_operations(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}\n", encoding="utf-8")
    api = MockApi()
    storage = HuggingFaceStorage(repo_id="org/repo", api=api)
    remote_manifest = storage.upload_bundle(bundle, "run/protein/pair/n_16")
    downloaded = storage.download_file("run/protein/pair/n_16/manifest.json", tmp_path / "downloads")
    downloaded_dir = storage.download_prefix("run/protein/pair/n_16", tmp_path / "bundle_download")
    assert remote_manifest.endswith("manifest.json")
    assert downloaded.is_file()
    assert (downloaded_dir / "manifest.json").is_file()
    assert (downloaded_dir / "nodes" / "sample_000.pdb").is_file()
    assert api.uploaded == [(str(bundle.resolve()), "run/protein/pair/n_16")]
    assert api.downloaded == [
        "run/protein/pair/n_16/manifest.json",
        "run/protein/pair/n_16/manifest.json",
        "run/protein/pair/n_16/nodes/sample_000.pdb",
    ]


def test_upload_file_is_verified(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    marker = tmp_path / "marker.json"
    marker.write_text("{}\n", encoding="utf-8")
    api = MockApi()
    storage = HuggingFaceStorage(repo_id="org/repo", api=api)
    remote = storage.upload_file(marker, "run/protein/pair/n_16/_worker_completed.json")
    assert remote.endswith("_worker_completed.json")
    assert api.uploaded_files == [(str(marker.resolve()), remote)]


def test_download_prefix_does_not_leave_partial_output_on_failure(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HF_TOKEN", "secret-token")

    class FailingApi(MockApi):
        def hf_hub_download(self, *, repo_id, repo_type, filename, local_dir):
            if filename.endswith("sample_000.pdb"):
                raise RuntimeError("download interrupted")
            return super().hf_hub_download(
                repo_id=repo_id, repo_type=repo_type, filename=filename, local_dir=local_dir
            )

    output = tmp_path / "bundle"
    output.mkdir()
    (output / "old.txt").write_text("keep\n", encoding="utf-8")
    storage = HuggingFaceStorage(repo_id="org/repo", api=FailingApi())

    with pytest.raises(RuntimeError, match="interrupted"):
        storage.download_prefix("run/protein/pair/n_16", output)

    assert (output / "old.txt").read_text(encoding="utf-8") == "keep\n"
    assert not (output / "manifest.json").exists()


def test_download_prefix_rolls_back_when_atomic_commit_fails(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    output = tmp_path / "bundle"
    output.mkdir()
    (output / "old.txt").write_text("keep\n", encoding="utf-8")
    storage = HuggingFaceStorage(repo_id="org/repo", api=MockApi())

    import parallel_bioemu_tools.hf_storage as storage_module

    original_replace = storage_module.os.replace
    calls = 0

    def fail_on_commit(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("commit failed")
        return original_replace(source, destination)

    monkeypatch.setattr(storage_module.os, "replace", fail_on_commit)
    with pytest.raises(OSError, match="commit failed"):
        storage.download_prefix("run/protein/pair/n_16", output)

    assert (output / "old.txt").read_text(encoding="utf-8") == "keep\n"
    assert not (output / "manifest.json").exists()
