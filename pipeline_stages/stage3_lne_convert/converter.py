from __future__ import annotations

import re
import shlex

from .backend.base import BackendCommandResult, RemoteBackend
from .config import Stage3Config


class Stage3Converter:
    FATAL_PATTERNS = [
        r"build error",
        r"permission denied",
        r"no such file or directory",
        r"\bnot found\b",
        r"make:\s*\*\*\*",
        r"converter failure",
        r"mapping failure",
        r"unsupported fatal error",
    ]
    WARNING_PATTERNS = [r"fatal:\s*not a git repository"]

    def run(self, backend: RemoteBackend, config: Stage3Config) -> BackendCommandResult:
        return backend.run_command(config.build_compile_command(), timeout=config.compile_timeout)

    def find_build_artifacts(self, backend: RemoteBackend, config: Stage3Config, model_name: str | None = None) -> BackendCommandResult:
        quoted_output_dir = shlex.quote(config.remote_output_dir)
        if model_name:
            command = (
                f"find {quoted_output_dir} -path '*{model_name}*' -type d -name 'lne_model_*'"
            )
        else:
            command = f"find {quoted_output_dir} -type d -name 'lne_model_*'"
        result = backend.run_command(command, timeout=config.compile_timeout)
        if model_name and (not result.stdout.strip()) and not result.error_type:
            return backend.run_command(f"find {quoted_output_dir} -type d -name 'lne_model_*'", timeout=config.compile_timeout)
        return result

    def evaluate_result(self, result: BackendCommandResult, artifact_paths: list[str]) -> dict:
        combined_log = "\n".join([result.stdout or "", result.stderr or ""])
        warning_matches = [pattern for pattern in self.WARNING_PATTERNS if re.search(pattern, combined_log, re.IGNORECASE)]
        fatal_matches = []
        sanitized_log = combined_log
        for warning_pattern in self.WARNING_PATTERNS:
            sanitized_log = re.sub(warning_pattern, "", sanitized_log, flags=re.IGNORECASE)
        for pattern in self.FATAL_PATTERNS:
            if re.search(pattern, sanitized_log, re.IGNORECASE):
                fatal_matches.append(pattern)

        if result.error_type in {"SSH_AUTH_FAILED", "WORKER_UNREACHABLE", "COMMAND_TIMEOUT", "SSH_BINARY_NOT_FOUND"}:
            conversion_status = result.error_type
            build_status = "NOT_RUN"
        elif result.exit_code != 0:
            conversion_status = "FAIL"
            build_status = "FAIL"
        elif fatal_matches:
            conversion_status = "FAIL"
            build_status = "FAIL"
        else:
            conversion_status = "SUCCESS"
            build_status = "SUCCESS" if artifact_paths else "FAIL"

        return {
            "command": result.command,
            "command_exit_code": result.exit_code,
            "command_error_type": result.error_type,
            "warning_matches": warning_matches,
            "fatal_matches": fatal_matches,
            "artifact_paths": artifact_paths,
            "artifact_count": len(artifact_paths),
            "conversion_status": conversion_status,
            "build_status": build_status,
        }

    @staticmethod
    def classify(result: BackendCommandResult) -> str:
        return "SUCCESS" if result.exit_code == 0 else "FAIL"
