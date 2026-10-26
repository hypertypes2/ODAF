from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BackendCommandResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    error_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class RemoteBackend(ABC):
    @abstractmethod
    def ensure_dir(self, remote_path: str) -> BackendCommandResult:
        raise NotImplementedError

    @abstractmethod
    def run_command(self, command: str, timeout: int | None = None) -> BackendCommandResult:
        raise NotImplementedError

    @abstractmethod
    def upload_file(self, local_path: str, remote_path: str) -> BackendCommandResult:
        raise NotImplementedError

    @abstractmethod
    def download_file(self, remote_path: str, local_path: str) -> BackendCommandResult:
        raise NotImplementedError
