from __future__ import annotations

import base64
import json
import sqlite3
import subprocess
from pathlib import Path

import pytest

import parallel_bioemu_tools.vscode_colab as colab_module
from parallel_bioemu_tools.vscode_colab import (
    AssignedColabServer,
    ColabJupyterClient,
    ColabConnectionError,
    load_assigned_colab_servers,
    read_vscode_safe_storage_password,
    select_assigned_server,
    wait_for_assigned_server,
)


def test_keychain_password_is_read_once_and_newline_is_removed() -> None:
    calls: list[list[str]] = []

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        del kwargs
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=b"secret\n", stderr=b"")

    assert read_vscode_safe_storage_password(runner) == b"secret"
    assert len(calls) == 1
    assert calls[0][:3] == ["security", "find-generic-password", "-w"]


def test_load_and_select_l4_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "state.vscdb"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE ItemTable (key TEXT, value TEXT)")
        connection.execute(
            "INSERT INTO ItemTable VALUES (?, ?)",
            (
                colab_module.ASSIGNED_SERVERS_KEY,
                json.dumps({"type": "Buffer", "data": [118, 49, 48]}),
            ),
        )
    decrypted = json.dumps(
        [
            {
                "label": "Colab L4",
                "accelerator": "GPU",
                "variant": "L4",
                "connectionInformation": {
                    "baseUrl": "https://example.test/",
                    "token": "token",
                    "headers": {"X-Colab": "header"},
                },
            }
        ]
    ).encode()
    monkeypatch.setattr(colab_module, "decrypt_vscode_secret", lambda value, password: decrypted)

    servers = load_assigned_colab_servers(b"password", database)
    selected = select_assigned_server(servers, "L4")

    assert selected.base_url == "https://example.test"
    assert selected.variant == "L4"
    assert selected.headers == {"X-Colab": "header"}


def test_select_server_rejects_wrong_gpu() -> None:
    server = AssignedColabServer(
        base_url="https://example.test",
        token="",
        headers={},
        accelerator="GPU",
        variant="T4",
        label="Colab T4",
    )

    with pytest.raises(ColabConnectionError, match="No active L4"):
        select_assigned_server([server], "L4")


def test_wait_for_server_reloads_extension_state() -> None:
    server = AssignedColabServer(
        base_url="https://example.test",
        token="",
        headers={},
        accelerator="GPU",
        variant="L4",
        label="Colab L4",
    )
    responses = iter([[], [server]])

    selected = wait_for_assigned_server(
        b"password",
        "L4",
        timeout_seconds=1,
        poll_seconds=0.001,
        server_loader=lambda password, database: next(responses),
    )

    assert selected is server


def test_wait_for_existing_kernel_to_become_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = AssignedColabServer(
        base_url="https://example.test",
        token="",
        headers={},
        accelerator="GPU",
        variant="L4",
        label="Colab L4",
    )
    client = ColabJupyterClient(server)
    states = iter(
        [
            [{"id": "kernel-1", "execution_state": "busy"}],
            [{"id": "kernel-1", "execution_state": "idle"}],
        ]
    )
    monkeypatch.setattr(client, "list_kernels", lambda: next(states))

    assert client.wait_for_kernel_idle(
        "kernel-1", timeout_seconds=1, poll_seconds=0.001
    )


def test_download_closes_response_when_status_check_fails(monkeypatch, tmp_path) -> None:
    server = AssignedColabServer(
        base_url="https://example.test", token="", headers={}, accelerator="GPU", variant="L4", label="Colab L4"
    )
    client = ColabJupyterClient(server)

    class Response:
        status_code = 500
        ok = False
        closed = False

        def iter_content(self, **kwargs):
            return iter(())

        def close(self):
            self.closed = True

        def json(self):
            return {}

    response = Response()
    monkeypatch.setattr(client, "request", lambda *args, **kwargs: response)
    with pytest.raises(ColabConnectionError):
        client.download("/content/result.zip", tmp_path / "result.zip")
    assert response.closed


def test_download_uses_contents_api_after_transient_file_404(monkeypatch, tmp_path) -> None:
    server = AssignedColabServer(
        base_url="https://example.test",
        token="",
        headers={},
        accelerator="GPU",
        variant="T4",
        label="Colab T4",
    )
    client = ColabJupyterClient(server)

    class Response:
        def __init__(self, status_code: int, payload: dict[str, str] | None = None) -> None:
            self.status_code = status_code
            self.ok = status_code < 400
            self.payload = payload or {}
            self.closed = False

        def iter_content(self, **kwargs):
            del kwargs
            return iter(())

        def close(self):
            self.closed = True

        def json(self):
            return self.payload

    missing = Response(404)
    contents = Response(
        200,
        {
            "type": "file",
            "format": "base64",
            "content": base64.b64encode(b"completed archive").decode("ascii"),
        },
    )
    responses = iter([missing, contents])
    calls: list[str] = []

    def request(method, path, **kwargs):
        del method, kwargs
        calls.append(path)
        return next(responses)

    monkeypatch.setattr(client, "request", request)
    destination = client.download("/content/exports/result.zip", tmp_path / "result.zip")

    assert destination.read_bytes() == b"completed archive"
    assert calls == ["files/content/exports/result.zip", "api/contents/content/exports/result.zip"]
    assert missing.closed
    assert contents.closed
