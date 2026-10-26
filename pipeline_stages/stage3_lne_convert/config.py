from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationCriteria:
    max_abs_error: float | None = None
    allow_cpu_fallback: bool | None = None
    qpoint: int | None = 10
    expected_worst_delta: float | None = 0.005859375


@dataclass
class Stage3Config:
    ssh_host: str = "heetae-chung-lne-odaf.bee0.lge.com"
    ssh_user: str = "worker"
    ssh_port: int = 59349
    ssh_key_path: str = ""
    remote_ddk_root: str = "/home/worker/lne_ddk-lneddk_3.0-2609"
    remote_tflite_dir: str = "/home/worker/lne_ddk-lneddk_3.0-2609/tflite"
    remote_raw_input_dir: str = "/home/worker/lne_ddk-lneddk_3.0-2609/raw_input"
    remote_output_dir: str = "/home/worker/lne_ddk-lneddk_3.0-2609/output"
    remote_lne_json_path: str = "/home/worker/lne_ddk-lneddk_3.0-2609/lne.json"
    lne_target: str = "dqc"
    validation_set: int = 10
    quant_loop: int = 10
    compile_timeout: int = 900
    spike_timeout: int = 900
    execution_mode: str = "dry-run"
    remote_lne_command: str = ""
    remote_spike_command: str = ""
    retry_count: int = 1
    validation: ValidationCriteria = field(default_factory=ValidationCriteria)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None = None) -> "Stage3Config":
        mapping = mapping or {}
        validation_mapping = mapping.get("validation", {})
        remote_ddk_root = mapping.get("remote_ddk_root") or os.getenv("ODAF_LNE_DDK_ROOT") or os.getenv("ODAF_VM_WORKSPACE") or cls.remote_ddk_root
        return cls(
            ssh_host=mapping.get("ssh_host") or os.getenv("ODAF_VM_HOST", cls.ssh_host),
            ssh_user=mapping.get("ssh_user") or mapping.get("ssh_username") or os.getenv("ODAF_VM_USER", cls.ssh_user),
            ssh_port=int(mapping.get("ssh_port") or os.getenv("ODAF_VM_PORT", cls.ssh_port)),
            ssh_key_path=mapping.get("ssh_key_path") or os.getenv("ODAF_VM_KEY_PATH", ""),
            remote_ddk_root=remote_ddk_root,
            remote_tflite_dir=mapping.get("remote_tflite_dir") or f"{remote_ddk_root}/tflite",
            remote_raw_input_dir=mapping.get("remote_raw_input_dir") or f"{remote_ddk_root}/raw_input",
            remote_output_dir=mapping.get("remote_output_dir") or f"{remote_ddk_root}/output",
            remote_lne_json_path=mapping.get("remote_lne_json_path") or f"{remote_ddk_root}/lne.json",
            lne_target=mapping.get("lne_target") or os.getenv("ODAF_LNE_TARGET", cls.lne_target),
            validation_set=int(mapping.get("validation_set") or os.getenv("ODAF_LNE_VALIDATION_SET", cls.validation_set)),
            quant_loop=int(mapping.get("quant_loop") or os.getenv("ODAF_LNE_QUANT_LOOP", cls.quant_loop)),
            compile_timeout=int(mapping.get("compile_timeout") or os.getenv("ODAF_LNE_COMPILE_TIMEOUT", cls.compile_timeout)),
            spike_timeout=int(mapping.get("spike_timeout") or os.getenv("ODAF_LNE_SPIKE_TIMEOUT", cls.spike_timeout)),
            execution_mode=mapping.get("execution_mode", cls.execution_mode),
            remote_lne_command=mapping.get("remote_lne_command") or os.getenv("ODAF_LNE_COMMAND", ""),
            remote_spike_command=mapping.get("remote_spike_command") or os.getenv("ODAF_SPIKE_COMMAND", ""),
            retry_count=int(mapping.get("retry_count", 1)),
            validation=ValidationCriteria(
                max_abs_error=validation_mapping.get("max_abs_error"),
                allow_cpu_fallback=validation_mapping.get("allow_cpu_fallback"),
                qpoint=int(validation_mapping.get("qpoint") or os.getenv("ODAF_LNE_QPOINT", 10)),
                expected_worst_delta=float(validation_mapping.get("expected_worst_delta") or os.getenv("ODAF_LNE_EXPECTED_WORST_DELTA", 0.005859375)),
            ),
        )

    @property
    def ssh_username(self) -> str:
        return self.ssh_user

    @property
    def remote_workspace(self) -> str:
        return self.remote_ddk_root

    def build_compile_command(self) -> str:
        if self.remote_lne_command:
            return self.remote_lne_command
        remote_json_name = os.path.basename(self.remote_lne_json_path)
        return f"cd {shlex.quote(self.remote_ddk_root)} && ./compile.sh -j {shlex.quote(remote_json_name)} -t {shlex.quote(self.lne_target)}"

    def build_spike_command(self) -> str:
        if self.remote_spike_command:
            return self.remote_spike_command
        remote_json_name = os.path.basename(self.remote_lne_json_path)
        return f"cd {shlex.quote(self.remote_ddk_root)} && ./run_spike.sh --net ./{shlex.quote(remote_json_name)}"

    def to_ddk_relative_path(self, absolute_path: str) -> str:
        if absolute_path.startswith(self.remote_ddk_root):
            relative = absolute_path[len(self.remote_ddk_root):]
            return relative if relative.startswith("/") else f"/{relative}"
        return absolute_path

    def missing_required_fields(self) -> list[str]:
        missing = []
        if not self.ssh_host:
            missing.append("ssh_host")
        if not self.ssh_user:
            missing.append("ssh_user")
        if not self.remote_ddk_root:
            missing.append("remote_ddk_root")
        if not self.remote_tflite_dir:
            missing.append("remote_tflite_dir")
        if not self.remote_raw_input_dir:
            missing.append("remote_raw_input_dir")
        if not self.remote_output_dir:
            missing.append("remote_output_dir")
        if not self.remote_lne_json_path:
            missing.append("remote_lne_json_path")
        return missing
