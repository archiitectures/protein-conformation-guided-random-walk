"""Storage abstraction for Hugging Face Hub uploads and downloads."""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from pathlib import PurePosixPath


class HfStorageError(RuntimeError):
    """Raised when storage operations fail."""


class HfApiProtocol(Protocol):
    """Minimal subset of the Hugging Face API used by the worker and collector."""

    def upload_folder(
        self,
        *,
        repo_id: str,
        repo_type: str,
        folder_path: str,
        path_in_repo: str,
        ignore_patterns: list[str] | None = None,
    ) -> Any:
        ...

    def upload_file(
        self,
        *,
        repo_id: str,
        repo_type: str,
        path_or_fileobj: str,
        path_in_repo: str,
    ) -> Any:
        ...

    def hf_hub_download(
        self,
        *,
        repo_id: str,
        repo_type: str,
        filename: str,
        local_dir: str,
    ) -> str:
        ...

    def list_repo_files(
        self,
        *,
        repo_id: str,
        repo_type: str,
    ) -> list[str]:
        ...


def resolve_hf_token(explicit_token: str | None = None) -> str:
    """Return a token from an explicit secure input or the environment."""
    token = explicit_token or os.environ.get("HF_TOKEN")
    if not token:
        raise HfStorageError("HF_TOKEN is required for Hugging Face operations")
    return token


@dataclass(slots=True)
class HuggingFaceStorage:
    """Small wrapper around an injectable HF backend."""

    repo_id: str
    api: HfApiProtocol | None = None
    token: str | None = None
    repo_type: str = "dataset"
    _token: str = field(init=False, repr=False)
    _io_lock: threading.RLock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.repo_id.strip():
            raise ValueError("repo_id must be non-empty")
        self._token = resolve_hf_token(self.token)
        self._io_lock = threading.RLock()
        if self.api is None:
            self.api = self._build_default_api(self._token)

    @staticmethod
    def _build_default_api(token: str) -> HfApiProtocol:
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except ImportError as exc:  # pragma: no cover - optional dependency.
            raise HfStorageError(
                "huggingface_hub is not installed; add the optional dependency "
                "to use live HF storage"
            ) from exc

        class _DefaultApi:
            def __init__(self, resolved_token: str) -> None:
                self._api = HfApi(token=resolved_token)

            def upload_folder(
                self,
                *,
                repo_id: str,
                repo_type: str,
                folder_path: str,
                path_in_repo: str,
                ignore_patterns: list[str] | None = None,
            ) -> Any:
                return self._api.upload_folder(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    folder_path=folder_path,
                    path_in_repo=path_in_repo,
                    ignore_patterns=ignore_patterns,
                )

            def hf_hub_download(
                self,
                *,
                repo_id: str,
                repo_type: str,
                filename: str,
                local_dir: str,
            ) -> str:
                return hf_hub_download(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    filename=filename,
                    local_dir=local_dir,
                    token=resolved_token,
                )

            def upload_file(
                self,
                *,
                repo_id: str,
                repo_type: str,
                path_or_fileobj: str,
                path_in_repo: str,
            ) -> Any:
                return self._api.upload_file(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    path_or_fileobj=path_or_fileobj,
                    path_in_repo=path_in_repo,
                )

            def list_repo_files(
                self,
                *,
                repo_id: str,
                repo_type: str,
            ) -> list[str]:
                return list(self._api.list_repo_files(repo_id=repo_id, repo_type=repo_type))

        resolved_token = token
        return _DefaultApi(resolved_token)

    def upload_bundle(self, local_dir: str | Path, remote_prefix: str) -> str:
        """Upload one local bundle directory and verify every regular file."""
        with self._io_lock:
            folder = Path(local_dir).expanduser().resolve()
            if not folder.is_dir():
                raise FileNotFoundError(f"Upload directory not found: {folder}")
            local_files = [path for path in folder.rglob("*") if path.is_file() and path.name not in {"_worker_completed.json", "_chunk_completed.json"}]
            symlinks = [path for path in folder.rglob("*") if path.is_symlink()]
            if symlinks:
                raise HfStorageError(f"Refusing to upload symlinks from bundle: {symlinks[0]}")
            if not local_files:
                raise HfStorageError(f"Refusing to upload empty bundle: {folder}")
            assert self.api is not None
            self.api.upload_folder(repo_id=self.repo_id, repo_type=self.repo_type, folder_path=str(folder),
                                   path_in_repo=remote_prefix,
                                   ignore_patterns=["_worker_completed.json", "_chunk_completed.json"])
            expected = [f"{remote_prefix.rstrip('/')}/{path.relative_to(folder).as_posix()}" for path in local_files]
            self.verify_remote_files(expected)
            return f"{remote_prefix.rstrip('/')}/manifest.json"

    def upload_file(self, local_path: str | Path, remote_path: str) -> str:
        """Upload and remotely verify one file."""
        with self._io_lock:
            source = Path(local_path).expanduser().resolve()
            if not source.is_file() or source.is_symlink():
                raise FileNotFoundError(f"Upload file not found or unsafe: {source}")
            assert self.api is not None
            self.api.upload_file(repo_id=self.repo_id, repo_type=self.repo_type,
                                 path_or_fileobj=str(source), path_in_repo=remote_path)
            self.verify_remote_files([remote_path])
            return remote_path

    def download_file(self, remote_path: str, local_dir: str | Path) -> Path:
        """Download one repo file into ``local_dir``."""
        with self._io_lock:
            output_dir = Path(local_dir).expanduser().resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            assert self.api is not None
            downloaded = self.api.hf_hub_download(repo_id=self.repo_id, repo_type=self.repo_type,
                                                  filename=remote_path, local_dir=str(output_dir))
            result = Path(downloaded).expanduser().resolve()
            if not result.is_file():
                raise HfStorageError(f"Hugging Face download did not produce a file: {remote_path}")
            return result

    def list_files(self) -> list[str]:
        """List tracked files for the backing repo."""
        with self._io_lock:
            assert self.api is not None
            return self.api.list_repo_files(repo_id=self.repo_id, repo_type=self.repo_type)

    def verify_remote_files(self, remote_paths: list[str]) -> None:
        """Raise unless every expected path is visible in the remote repository."""
        tracked = set(self.list_files())
        missing = sorted(set(remote_paths) - tracked)
        if missing:
            preview = ", ".join(missing[:3])
            raise HfStorageError(f"Hugging Face upload verification failed; missing: {preview}")

    def download_prefix(self, remote_prefix: str, local_dir: str | Path) -> Path:
        """Download all files under one repo prefix into ``local_dir``."""
        with self._io_lock:
            output_dir = Path(local_dir).expanduser().resolve()
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            prefix = remote_prefix.rstrip("/") + "/"
            matched = [path for path in self.list_files() if path.startswith(prefix)]
            if not matched:
                raise FileNotFoundError(f"No Hugging Face files found under prefix: {remote_prefix}")
            with tempfile.TemporaryDirectory(prefix="hf-download-", dir=output_dir.parent) as temp_dir:
                staging = Path(temp_dir) / "payload"
                staging.mkdir()
                seen_relative: set[tuple[str, ...]] = set()
                for remote_path in matched:
                    relative_path = PurePosixPath(remote_path).relative_to(PurePosixPath(prefix))
                    if (not relative_path.parts or relative_path.is_absolute()
                            or ".." in relative_path.parts or "\\" in remote_path):
                        raise HfStorageError(f"Unsafe Hugging Face repository path: {remote_path}")
                    if relative_path.parts in seen_relative:
                        raise HfStorageError(f"Duplicate Hugging Face repository path: {remote_path}")
                    seen_relative.add(relative_path.parts)
                    downloaded = self.download_file(remote_path, Path(temp_dir) / "files")
                    destination = staging.joinpath(*relative_path.parts)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(downloaded, destination)
                backup: Path | None = None
                try:
                    if output_dir.exists():
                        backup = Path(tempfile.mkdtemp(
                            prefix=f".{output_dir.name}.backup-", dir=output_dir.parent
                        ))
                        shutil.rmtree(backup)
                        os.replace(output_dir, backup)
                    os.replace(staging, output_dir)
                except Exception:
                    if output_dir.exists() and backup is not None and backup.exists():
                        shutil.rmtree(output_dir)
                    if backup is not None and backup.exists():
                        os.replace(backup, output_dir)
                    raise
                finally:
                    if backup is not None and backup.exists():
                        shutil.rmtree(backup)
            return output_dir
