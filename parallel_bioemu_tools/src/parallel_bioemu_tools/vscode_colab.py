"""Authenticated access to a VS Code Colab extension runtime on macOS."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import sqlite3
import ssl
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, urlparse

import certifi
import requests
from tornado.httpclient import HTTPRequest
from tornado.httputil import HTTPHeaders
from tornado.websocket import WebSocketClosedError, websocket_connect


VS_CODE_DATABASE = (
    Path.home() / "Library/Application Support/Code/User/globalStorage/state.vscdb"
)
ASSIGNED_SERVERS_KEY = (
    'secret://{"extensionId":"google.colab","key":"google.assigned_servers"}'
)
KEYCHAIN_SERVICE = "Code Safe Storage"
KEYCHAIN_ACCOUNT = "Code"


class ColabConnectionError(RuntimeError):
    """Raised when the assigned Colab server cannot be loaded or contacted."""


@dataclass(frozen=True, slots=True)
class AssignedColabServer:
    """One assigned server record from the VS Code Colab extension."""

    base_url: str
    token: str
    headers: dict[str, str]
    accelerator: str
    variant: str
    label: str

    @property
    def gpu_description(self) -> str:
        """Return the normalized accelerator fields used for admission."""
        return " ".join((self.accelerator, self.variant, self.label)).upper()


def read_vscode_safe_storage_password(
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> bytes:
    """Read the VS Code safe-storage key from macOS Keychain once in memory.

    macOS may display one authorization prompt. The returned bytes are never
    persisted by this module and callers should keep their scope short.
    """
    try:
        result = runner(
            [
                "security",
                "find-generic-password",
                "-w",
                "-s",
                KEYCHAIN_SERVICE,
                "-a",
                KEYCHAIN_ACCOUNT,
            ],
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ColabConnectionError(
            "Could not read the VS Code safe-storage key from macOS Keychain"
        ) from exc
    return result.stdout.rstrip(b"\n")


def decrypt_vscode_secret(value: bytes, password: bytes) -> bytes:
    """Decrypt an Electron ``safeStorage`` v10 value using an in-memory key."""
    if not value.startswith(b"v10"):
        raise ColabConnectionError("Unexpected VS Code safe-storage secret format")
    key = hashlib.pbkdf2_hmac("sha1", password, b"saltysalt", 1003, 16)
    try:
        result = subprocess.run(
            [
                "openssl",
                "enc",
                "-aes-128-cbc",
                "-d",
                "-K",
                key.hex(),
                "-iv",
                (b" " * 16).hex(),
            ],
            input=value[3:],
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ColabConnectionError("Could not decrypt the VS Code Colab record") from exc
    return result.stdout


def load_assigned_colab_servers(
    password: bytes,
    database_path: str | Path = VS_CODE_DATABASE,
) -> list[AssignedColabServer]:
    """Load and decrypt the Colab extension's assigned-server records."""
    database = Path(database_path).expanduser()
    if not database.is_file():
        raise ColabConnectionError(f"VS Code state database not found: {database}")
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT value FROM ItemTable WHERE key = ?",
            (ASSIGNED_SERVERS_KEY,),
        ).fetchone()
    if row is None:
        raise ColabConnectionError("VS Code has no stored Colab assigned-server record")

    raw = row[0]
    if isinstance(raw, str) and raw.startswith('{"type":"Buffer"'):
        encrypted = bytes(json.loads(raw)["data"])
    elif isinstance(raw, bytes):
        encrypted = raw
    else:
        encrypted = str(raw).encode("latin1")

    try:
        decoded: Any = json.loads(decrypt_vscode_secret(encrypted, password))
        if isinstance(decoded, str):
            decoded = json.loads(decoded)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ColabConnectionError("Decrypted Colab server record is not valid JSON") from exc
    records = list(decoded.values()) if isinstance(decoded, dict) else decoded
    if not isinstance(records, list):
        raise ColabConnectionError("Colab assigned-server record has an unexpected shape")

    servers: list[AssignedColabServer] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        connection_info = record.get("connectionInformation")
        if not isinstance(connection_info, dict) or not connection_info.get("baseUrl"):
            continue
        raw_headers = connection_info.get("headers") or {}
        servers.append(
            AssignedColabServer(
                base_url=str(connection_info["baseUrl"]).rstrip("/"),
                token=str(connection_info.get("token") or ""),
                headers={str(key): str(value) for key, value in raw_headers.items()},
                accelerator=str(record.get("accelerator") or ""),
                variant=str(record.get("variant") or ""),
                label=str(record.get("label") or ""),
            )
        )
    return servers


def select_assigned_server(
    servers: list[AssignedColabServer],
    required_gpu: str,
) -> AssignedColabServer:
    """Select the first assigned server whose metadata names ``required_gpu``."""
    required = required_gpu.strip().upper()
    if not required:
        raise ValueError("required_gpu must be non-empty")
    for server in servers:
        if required in server.gpu_description:
            return server
    available = [server.gpu_description or "UNKNOWN" for server in servers]
    raise ColabConnectionError(
        f"No active {required} Colab assignment was found; assigned servers: {available}"
    )


def wait_for_assigned_server(
    password: bytes,
    required_gpu: str,
    *,
    database_path: str | Path = VS_CODE_DATABASE,
    timeout_seconds: float = 15 * 60,
    poll_seconds: float = 5,
    server_loader: Callable[[bytes, str | Path], list[AssignedColabServer]] = (
        load_assigned_colab_servers
    ),
) -> AssignedColabServer:
    """Poll the extension state until the requested GPU assignment appears."""
    if timeout_seconds < 0:
        raise ValueError("timeout_seconds must be non-negative")
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    deadline = time.monotonic() + timeout_seconds
    while True:
        servers = server_loader(password, database_path)
        try:
            return select_assigned_server(servers, required_gpu)
        except ColabConnectionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))


class ColabJupyterClient:
    """Minimal authenticated Jupyter REST and WebSocket client for Colab."""

    def __init__(self, server: AssignedColabServer) -> None:
        self.server = server
        self.session = requests.Session()
        self.session.headers.update(server.headers)

    def request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        """Issue an authenticated request without exposing credentials in errors."""
        params = dict(kwargs.pop("params", {}) or {})
        if self.server.token:
            params.setdefault("token", self.server.token)
        try:
            response = self.session.request(
                method,
                f"{self.server.base_url}/{path.lstrip('/')}",
                params=params,
                timeout=kwargs.pop("timeout", 60),
                **kwargs,
            )
        except requests.RequestException as exc:
            raise ColabConnectionError(
                f"Colab request failed for {method} {path}: {type(exc).__name__}"
            ) from exc
        return response

    def status(self) -> dict[str, Any]:
        """Return Jupyter status metadata after validating connectivity."""
        response = self.request("GET", "api/status", timeout=30)
        self._raise_for_status(response, "status")
        value = response.json()
        return value if isinstance(value, dict) else {}

    def upload(self, source: str | Path, remote_name: str) -> None:
        """Upload a local file through the Jupyter Contents API."""
        path = Path(source)
        payload = {
            "type": "file",
            "format": "base64",
            "content": base64.b64encode(path.read_bytes()).decode("ascii"),
        }
        response = self.request(
            "PUT",
            f"api/contents/{quote(remote_name.lstrip('/'))}",
            json=payload,
            timeout=600,
        )
        self._raise_for_status(response, f"upload {remote_name}")

    def list_kernels(self) -> list[dict[str, Any]]:
        """Return active Jupyter kernels."""
        response = self.request("GET", "api/kernels")
        self._raise_for_status(response, "list kernels")
        value = response.json()
        return value if isinstance(value, list) else []

    def create_kernel(self) -> str:
        """Create and return a dedicated Python kernel ID."""
        response = self.request("POST", "api/kernels", json={"name": "python3"})
        self._raise_for_status(response, "create kernel")
        kernel_id = str(response.json().get("id") or "")
        if not kernel_id:
            raise ColabConnectionError("Colab did not return a kernel ID")
        return kernel_id

    def wait_for_kernel_idle(
        self,
        kernel_id: str,
        *,
        timeout_seconds: float = 4 * 60 * 60,
        poll_seconds: float = 60,
    ) -> bool:
        """Wait for a prior controller kernel, returning false if it is gone."""
        deadline = time.monotonic() + timeout_seconds
        while True:
            kernel = next(
                (item for item in self.list_kernels() if str(item.get("id")) == kernel_id),
                None,
            )
            if kernel is None:
                print(f"Recovery kernel is no longer active: {kernel_id}", flush=True)
                return False
            state = str(kernel.get("execution_state") or "unknown")
            if state == "idle":
                print(f"Recovery kernel is idle: {kernel_id}", flush=True)
                return True
            if time.monotonic() >= deadline:
                raise ColabConnectionError(
                    f"Recovery kernel did not become idle within {timeout_seconds:g} seconds"
                )
            time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))

    def remote_file_exists(self, remote_path: str) -> bool:
        """Return whether a remote Jupyter file exists without downloading it."""
        response = self.request(
            "GET",
            f"api/contents/{quote(remote_path.lstrip('/'))}",
            params={"content": "0"},
        )
        if response.status_code == 404:
            return False
        self._raise_for_status(response, f"inspect {remote_path}")
        return True

    def download(self, remote_path: str, destination: str | Path) -> Path:
        """Stream a remote file to an atomic local destination."""
        output = Path(destination)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.name}.", suffix=".part", dir=output.parent, delete=False
        ) as temporary_handle:
            temporary = Path(temporary_handle.name)
        response = self.request(
            "GET",
            f"files/{quote(remote_path.lstrip('/'))}",
            stream=True,
            timeout=3600,
        )
        try:
            self._raise_for_status(response, f"download {remote_path}")
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
            response.close()
        return output

    def execute(
        self,
        kernel_id: str,
        code: str,
        label: str,
        *,
        allow_disconnect: bool = False,
        timeout_seconds: float = 8 * 60 * 60,
    ) -> str:
        """Execute code, stream textual output, and wait for the kernel to idle."""
        return asyncio.run(
            self._execute_async(
                kernel_id,
                code,
                label,
                allow_disconnect=allow_disconnect,
                timeout_seconds=timeout_seconds,
            )
        )

    async def _execute_async(
        self,
        kernel_id: str,
        code: str,
        label: str,
        *,
        allow_disconnect: bool,
        timeout_seconds: float,
    ) -> str:
        """WebSocket implementation for :meth:`execute`."""
        parsed = urlparse(self.server.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        session_id = uuid.uuid4().hex
        token_query = f"&token={quote(self.server.token)}" if self.server.token else ""
        url = (
            f"{scheme}://{parsed.netloc}{parsed.path.rstrip('/')}/api/kernels/"
            f"{quote(kernel_id)}/channels?session_id={session_id}{token_query}"
        )
        headers = HTTPHeaders(self.server.headers)
        headers["Origin"] = f"{parsed.scheme}://{parsed.netloc}"
        request = HTTPRequest(
            url,
            headers=headers,
            ssl_options=ssl.create_default_context(cafile=certifi.where()),
            connect_timeout=30,
            request_timeout=0,
        )
        socket = await websocket_connect(request, ping_interval=30, ping_timeout=120)
        msg_id = uuid.uuid4().hex
        message = {
            "header": {
                "msg_id": msg_id,
                "username": "guided-conformation-controller",
                "session": session_id,
                "date": datetime.now(timezone.utc).isoformat(),
                "msg_type": "execute_request",
                "version": "5.3",
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": True,
                "user_expressions": {},
                "allow_stdin": False,
                "stop_on_error": True,
            },
            "channel": "shell",
            "buffers": [],
        }
        print(f"\n=== {label} ===", flush=True)
        socket.write_message(json.dumps(message))
        output: list[str] = []
        error_text: str | None = None

        async def receive_until_idle() -> None:
            nonlocal error_text
            while True:
                raw_message = await socket.read_message()
                if raw_message is None:
                    raise WebSocketClosedError()
                if isinstance(raw_message, bytes):
                    raw_message = raw_message.decode("utf-8")
                message_data = json.loads(raw_message)
                if message_data.get("parent_header", {}).get("msg_id") != msg_id:
                    continue
                message_type = message_data.get("msg_type") or message_data.get(
                    "header", {}
                ).get("msg_type")
                content = message_data.get("content", {})
                text = ""
                if message_type == "stream":
                    text = str(content.get("text") or "")
                elif message_type in {"display_data", "execute_result"}:
                    text = str(content.get("data", {}).get("text/plain") or "")
                elif message_type == "error":
                    traceback_lines = content.get("traceback") or []
                    error_text = "\n".join(str(line) for line in traceback_lines)
                    if not error_text:
                        error_text = f"{content.get('ename')}: {content.get('evalue')}"
                if text:
                    print(text, end="" if text.endswith("\n") else "\n", flush=True)
                    output.append(text)
                if (
                    message_type == "status"
                    and content.get("execution_state") == "idle"
                ):
                    return

        try:
            await asyncio.wait_for(receive_until_idle(), timeout=timeout_seconds)
        except WebSocketClosedError:
            if not allow_disconnect or "RELEASE_REQUESTED" not in "".join(output):
                raise ColabConnectionError(
                    f"Colab WebSocket closed during {label}"
                ) from None
        finally:
            socket.close()
        if error_text:
            raise ColabConnectionError(f"Remote execution failed during {label}:\n{error_text}")
        return "".join(output)

    @staticmethod
    def _raise_for_status(response: requests.Response, action: str) -> None:
        if response.ok:
            return
        raise ColabConnectionError(
            f"Colab {action} returned HTTP {response.status_code}; "
            "response body omitted to avoid exposing credentials"
        )
