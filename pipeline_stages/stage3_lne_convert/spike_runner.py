from __future__ import annotations

import re
import shlex

from .backend.base import BackendCommandResult, RemoteBackend
from .config import Stage3Config


class Stage3SpikeRunner:
    FATAL_PATTERNS = [
        r"network description file.*not found",
        r"permission denied",
        r"no such file or directory",
        r"segmentation fault",
        r"traceback",
        r"assertion",
        r"error",
    ]

    def run(self, backend: RemoteBackend, config: Stage3Config) -> BackendCommandResult:
        return backend.run_command(config.build_spike_command(), timeout=config.spike_timeout)

    def discover_summary_paths(self, backend: RemoteBackend, config: Stage3Config, model_name: str | None = None) -> BackendCommandResult:
        quoted_output_dir = shlex.quote(config.remote_output_dir)
        if model_name:
            command = f"find {quoted_output_dir} -path '*{model_name}*' -type f -name summary.txt"
        else:
            command = f"find {quoted_output_dir} -type f -name summary.txt"
        result = backend.run_command(command, timeout=config.spike_timeout)
        if model_name and (not result.stdout.strip()) and not result.error_type:
            return backend.run_command(f"find {quoted_output_dir} -type f -name summary.txt", timeout=config.spike_timeout)
        return result

    def evaluate_result(self, result: BackendCommandResult, summary_paths: list[str], non_empty_count: int, expected_count: int) -> dict:
        combined_log = "\n".join([result.stdout or "", result.stderr or ""])
        fatal_matches = []
        for pattern in self.FATAL_PATTERNS:
            if re.search(pattern, combined_log, re.IGNORECASE) and "fatal: not a git repository" not in combined_log.lower():
                fatal_matches.append(pattern)

        summary_count = len(summary_paths)
        if result.error_type in {"SSH_AUTH_FAILED", "WORKER_UNREACHABLE", "COMMAND_TIMEOUT", "SSH_BINARY_NOT_FOUND"}:
            spike_status = result.error_type
        elif fatal_matches:
            spike_status = "FAIL"
        elif result.exit_code == 0 and summary_count >= expected_count and non_empty_count >= expected_count:
            spike_status = "SUCCESS"
        elif result.exit_code == 0 and summary_count > 0 and non_empty_count > 0:
            spike_status = "SUCCESS"
        else:
            spike_status = "FAIL"

        return {
            "command": result.command,
            "command_exit_code": result.exit_code,
            "command_error_type": result.error_type,
            "fatal_matches": fatal_matches,
            "summary_paths": summary_paths,
            "summary_count": summary_count,
            "non_empty_summary_count": non_empty_count,
            "expected_summary_count": expected_count,
            "spike_status": spike_status,
        }

    @staticmethod
    def classify(result: BackendCommandResult) -> str:
        return "SUCCESS" if result.exit_code == 0 else "FAIL"
