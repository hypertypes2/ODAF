from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass

from .base import BackendCommandResult, RemoteBackend


@dataclass
class SSHConnectionInfo:
    host: str
    username: str
    port: int = 22
    key_path: str = ""


class SSHLinuxVMBackend(RemoteBackend):
    AUTH_ERROR_PATTERNS = ["permission denied", "authentication failed"]
    UNREACHABLE_PATTERNS = [
        "could not resolve hostname",
        "name or service not known",
        "connection timed out",
        "connection refused",
        "no route to host",
        "connection closed by remote host",
        "connection closed",
    ]

    def __init__(self, connection: SSHConnectionInfo) -> None:
        self.connection = connection

    def _validate(self) -> None:
        if not self.connection.host or not self.connection.username:
            raise ValueError("SSH host와 username이 모두 필요합니다.")

    def _ssh_base_args(self) -> list[str]:
        self._validate()
        args = ["ssh", "-p", str(self.connection.port)]
        if self.connection.key_path:
            args.extend(["-i", self.connection.key_path])
        args.append(f"{self.connection.username}@{self.connection.host}")
        return args

    def _scp_base_args(self) -> list[str]:
        self._validate()
        args = ["scp", "-P", str(self.connection.port)]
        if self.connection.key_path:
            args.extend(["-i", self.connection.key_path])
        return args

    @classmethod
    def infer_error_type(cls, exit_code: int, stderr: str) -> str | None:
        stderr_lower = (stderr or "").lower()
        if any(pattern in stderr_lower for pattern in cls.AUTH_ERROR_PATTERNS):
            return "SSH_AUTH_FAILED"
        if any(pattern in stderr_lower for pattern in cls.UNREACHABLE_PATTERNS):
            return "WORKER_UNREACHABLE"
        if exit_code != 0 and stderr_lower:
            return "SSH_COMMAND_FAILED"
        return None

    def ensure_dir(self, remote_path: str) -> BackendCommandResult:
        return self.run_command(f"mkdir -p {shlex.quote(remote_path)}")

    def run_command(self, command: str, timeout: int | None = None) -> BackendCommandResult:
        try:
            process = subprocess.run(
                [*self._ssh_base_args(), command],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return BackendCommandResult(
                command=command,
                exit_code=process.returncode,
                stdout=process.stdout,
                stderr=process.stderr,
                error_type=self.infer_error_type(process.returncode, process.stderr),
            )
        except subprocess.TimeoutExpired as error:
            return BackendCommandResult(
                command=command,
                exit_code=-1,
                stdout=error.stdout or "",
                stderr=error.stderr or str(error),
                error_type="COMMAND_TIMEOUT",
            )
        except FileNotFoundError as error:
            return BackendCommandResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr=str(error),
                error_type="SSH_BINARY_NOT_FOUND",
            )

    def upload_file(self, local_path: str, remote_path: str) -> BackendCommandResult:
        target = f"{self.connection.username}@{self.connection.host}:{remote_path}"
        try:
            process = subprocess.run(
                [*self._scp_base_args(), local_path, target],
                capture_output=True,
                text=True,
                check=False,
            )
            return BackendCommandResult(
                command=f"scp {local_path} {target}",
                exit_code=process.returncode,
                stdout=process.stdout,
                stderr=process.stderr,
                error_type=self.infer_error_type(process.returncode, process.stderr),
            )
        except FileNotFoundError as error:
            return BackendCommandResult(
                command=f"scp {local_path} {target}",
                exit_code=-1,
                stdout="",
                stderr=str(error),
                error_type="SSH_BINARY_NOT_FOUND",
            )

    def download_file(self, remote_path: str, local_path: str) -> BackendCommandResult:
        source = f"{self.connection.username}@{self.connection.host}:{remote_path}"
        try:
            process = subprocess.run(
                [*self._scp_base_args(), source, local_path],
                capture_output=True,
                text=True,
                check=False,
            )
            return BackendCommandResult(
                command=f"scp {source} {local_path}",
                exit_code=process.returncode,
                stdout=process.stdout,
                stderr=process.stderr,
                error_type=self.infer_error_type(process.returncode, process.stderr),
            )
        except FileNotFoundError as error:
            return BackendCommandResult(
                command=f"scp {source} {local_path}",
                exit_code=-1,
                stdout="",
                stderr=str(error),
                error_type="SSH_BINARY_NOT_FOUND",
            )
