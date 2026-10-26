from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from google import genai
from google.genai import types

from .backend.base import BackendCommandResult
from .backend.ssh_linux_vm import SSHConnectionInfo, SSHLinuxVMBackend
from .config import Stage3Config
from .converter import Stage3Converter
from .result_parser import Stage3ResultParser
from .spike_runner import Stage3SpikeRunner
from .validator import Stage3Validator


load_dotenv()


class Stage3LNEConvertAgent:
    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.client = genai.Client(api_key=api_key) if api_key else None
        self.converter = Stage3Converter()
        self.spike_runner = Stage3SpikeRunner()
        self.parser = Stage3ResultParser()
        self.validator = Stage3Validator()

    def test_connection(self, config_mapping: dict[str, Any]) -> dict[str, Any]:
        config = Stage3Config.from_mapping(config_mapping)
        backend = self._build_backend(config)
        result = backend.run_command("pwd", timeout=15)
        status = "SUCCESS" if result.exit_code == 0 and not result.error_type else (result.error_type or "FAIL")
        return {"status": status, "exit_code": result.exit_code, "stdout": result.stdout, "stderr": result.stderr}

    def _build_backend(self, config: Stage3Config) -> SSHLinuxVMBackend:
        return SSHLinuxVMBackend(
            SSHConnectionInfo(
                host=config.ssh_host,
                username=config.ssh_username,
                port=config.ssh_port,
                key_path=config.ssh_key_path,
            )
        )

    def _make_run_dir(self, output_dir: str) -> tuple[str, str]:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(output_dir) / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summaries").mkdir(parents=True, exist_ok=True)
        return run_id, str(run_dir)

    def _derive_model_name(self, tflite_path: str) -> str:
        return Path(tflite_path).stem

    def _write_log(self, output_path: str | Path, content: str) -> str:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        return str(output_path)

    def _copy_to_compatibility_path(self, source_path: str, target_path: str | Path) -> str:
        target_path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target_path)
        return str(target_path)

    def _local_input_check(self, local_inputs: dict[str, str], config: Stage3Config) -> dict[str, Any]:
        errors: list[str] = []
        warnings: list[str] = []

        selected_tflite = local_inputs.get("selected_tflite") or local_inputs.get("int8_tflite") or local_inputs.get("fp32_tflite")
        if not selected_tflite:
            errors.append("selected_tflite is required")
            return {"errors": errors, "warnings": warnings}

        selected_tflite_path = Path(selected_tflite)
        if not selected_tflite_path.exists():
            errors.append("Selected TFLite file does not exist")
        elif selected_tflite_path.suffix.lower() != ".tflite":
            errors.append("Selected file is not a .tflite model")
        elif selected_tflite_path.stat().st_size == 0:
            errors.append("Selected TFLite file size is 0")

        raw_input_dir = Path(local_inputs.get("raw_input_dir", ""))
        raw_input_files = sorted(raw_input_dir.glob("*.bin")) if raw_input_dir.exists() else []
        if not raw_input_files:
            errors.append("No raw_input .bin files were found")

        raw_sizes = {file_path.stat().st_size for file_path in raw_input_files if file_path.exists()}
        if any(size == 0 for size in raw_sizes):
            errors.append("At least one raw_input file has size 0")
        if len(raw_sizes) > 1:
            errors.append("raw_input file sizes are inconsistent")
        if raw_input_files and config.validation_set > len(raw_input_files):
            errors.append("validation_set is larger than the available raw_input file count")

        reference_output_path = local_inputs.get("reference_output", "")
        if reference_output_path and not Path(reference_output_path).exists():
            warnings.append("reference_output path was provided but does not exist")

        model_name = self._derive_model_name(str(selected_tflite_path)) if selected_tflite_path.exists() else "model"

        return {
            "errors": errors,
            "warnings": warnings,
            "selected_tflite_path": str(selected_tflite_path),
            "reference_output_path": reference_output_path,
            "raw_input_files": [str(file_path) for file_path in raw_input_files],
            "raw_input_count": len(raw_input_files),
            "raw_input_size": next(iter(raw_sizes)) if raw_sizes else 0,
            "model_name": model_name,
        }

    def _build_lne_json_payload(self, config: Stage3Config, remote_model_filename: str) -> dict[str, Any]:
        return {
            "tflite": {
                "model_file": f"{config.to_ddk_relative_path(config.remote_tflite_dir)}/{remote_model_filename}",
                "input_dirs": [f"{config.to_ddk_relative_path(config.remote_raw_input_dir)}/"],
            },
            "input": [{"type": "raw", "input_type": "FLOAT32"}],
            "input_format": "NHWC",
            "output_format": "NHWC",
            "quant_loop": config.quant_loop,
            "validation_set": config.validation_set,
            "logistic": 1,
            "sw_act": 1,
            "max_lne_partitions": 2,
            "version": 2.0,
        }

    def _write_local_lne_json(self, run_dir: str, payload: dict[str, Any]) -> str:
        local_json_path = Path(run_dir) / "lne.json"
        local_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(local_json_path)

    def _backup_remote_file(self, backend: SSHLinuxVMBackend, remote_path: str, run_id: str) -> BackendCommandResult:
        backup_path = f"{remote_path}.bak.{run_id}"
        command = f"if [ -f {self._quote(remote_path)} ]; then cp {self._quote(remote_path)} {self._quote(backup_path)}; fi"
        return backend.run_command(command, timeout=30)

    def _backup_remote_raw_inputs(self, backend: SSHLinuxVMBackend, config: Stage3Config, run_id: str) -> BackendCommandResult:
        backup_dir = f"{config.remote_ddk_root}/raw_input_backup_{run_id}"
        command = (
            f"mkdir -p {self._quote(backup_dir)} && "
            f"if [ -d {self._quote(config.remote_raw_input_dir)} ]; then "
            f"find {self._quote(config.remote_raw_input_dir)} -maxdepth 1 -type f -name '*.bin' -exec mv {{}} {self._quote(backup_dir)}/ \\; ; fi"
        )
        return backend.run_command(command, timeout=60)

    def _upload_inputs(self, backend: SSHLinuxVMBackend, config: Stage3Config, run_id: str, local_check: dict[str, Any], local_lne_json_path: str) -> dict[str, Any]:
        uploads: list[dict[str, Any]] = []
        backend.ensure_dir(config.remote_tflite_dir)
        backend.ensure_dir(config.remote_raw_input_dir)
        backend.ensure_dir(config.remote_output_dir)
        backend.ensure_dir(f"{config.remote_output_dir}/reference")

        tflite_name = Path(local_check["selected_tflite_path"]).name
        remote_tflite_path = f"{config.remote_tflite_dir}/{tflite_name}"
        upload_tflite = backend.upload_file(local_check["selected_tflite_path"], remote_tflite_path)
        uploads.append({"kind": "tflite", "local": local_check["selected_tflite_path"], "remote": remote_tflite_path, "exit_code": upload_tflite.exit_code, "error_type": upload_tflite.error_type})

        raw_backup = self._backup_remote_raw_inputs(backend, config, run_id)
        uploads.append({"kind": "raw_input_backup", "remote": config.remote_raw_input_dir, "exit_code": raw_backup.exit_code, "error_type": raw_backup.error_type})

        for file_path in local_check["raw_input_files"]:
            remote_path = f"{config.remote_raw_input_dir}/{Path(file_path).name}"
            result = backend.upload_file(file_path, remote_path)
            uploads.append({"kind": "raw_input", "local": file_path, "remote": remote_path, "exit_code": result.exit_code, "error_type": result.error_type})

        lne_backup = self._backup_remote_file(backend, config.remote_lne_json_path, run_id)
        uploads.append({"kind": "lne_json_backup", "remote": config.remote_lne_json_path, "exit_code": lne_backup.exit_code, "error_type": lne_backup.error_type})
        upload_lne_json = backend.upload_file(local_lne_json_path, config.remote_lne_json_path)
        uploads.append({"kind": "lne_json", "local": local_lne_json_path, "remote": config.remote_lne_json_path, "exit_code": upload_lne_json.exit_code, "error_type": upload_lne_json.error_type})

        remote_reference_output_path = ""
        if local_check.get("reference_output_path") and Path(local_check["reference_output_path"]).exists():
            remote_reference_output_path = f"{config.remote_output_dir}/reference/{Path(local_check['reference_output_path']).name}"
            upload_reference = backend.upload_file(local_check["reference_output_path"], remote_reference_output_path)
            uploads.append({"kind": "reference_output", "local": local_check["reference_output_path"], "remote": remote_reference_output_path, "exit_code": upload_reference.exit_code, "error_type": upload_reference.error_type})

        return {"uploads": uploads, "remote_tflite_path": remote_tflite_path, "remote_reference_output_path": remote_reference_output_path}

    def _discover_artifact_paths(self, backend: SSHLinuxVMBackend, config: Stage3Config, model_name: str) -> list[str]:
        result = self.converter.find_build_artifacts(backend, config, model_name)
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def _discover_summary_paths(self, backend: SSHLinuxVMBackend, config: Stage3Config, model_name: str) -> list[str]:
        result = self.spike_runner.discover_summary_paths(backend, config, model_name)
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def _download_summary_files(self, backend: SSHLinuxVMBackend, summary_paths: list[str], run_dir: str) -> list[dict[str, Any]]:
        summaries_dir = Path(run_dir) / "summaries"
        downloaded: list[dict[str, Any]] = []
        for index, remote_path in enumerate(summary_paths):
            local_path = summaries_dir / f"summary_{index:02d}.txt"
            result = backend.download_file(remote_path, str(local_path))
            content = local_path.read_text(encoding="utf-8", errors="replace") if local_path.exists() else ""
            downloaded.append({"remote_path": remote_path, "local_path": str(local_path), "download_exit_code": result.exit_code, "download_error_type": result.error_type, "size": local_path.stat().st_size if local_path.exists() else 0, "content": content})
        return downloaded

    def _analyze_failure(self, source_code: str, facts: dict[str, Any], compile_log: str, spike_log: str) -> dict[str, Any]:
        if self.client:
            prompt = (
                "당신은 LNE DDK 및 SPIKE 로그를 해석하는 온디바이스 AI 에이전트입니다.\n"
                "facts와 agent_analysis를 구분해 JSON으로 응답하십시오.\n"
                "{\n"
                '  "facts": ["로그에서 직접 확인된 사실"],\n'
                '  "agent_analysis": "원인 해석",\n'
                '  "recommended_actions": ["조치1", "조치2"],\n'
                '  "needs_stage2_rebuild": true\n'
                "}\n\n"
                f"[SOURCE_CODE]\n{source_code}\n\n"
                f"[FACTS]\n{json.dumps(facts, ensure_ascii=False)}\n\n"
                f"[COMPILE_LOG]\n{compile_log}\n\n"
                f"[SPIKE_LOG]\n{spike_log}"
            )
            try:
                response = self.client.models.generate_content(model="gemini-3.6-flash", contents=prompt, config=types.GenerateContentConfig(response_mime_type="application/json"))
                return json.loads(response.text.strip())
            except Exception:
                pass

        simple_facts = [
            f"build_artifact_count={facts.get('artifact_count', 0)}",
            f"summary_count={facts.get('validation_summary_count', 0)}",
            f"max_final_output_abs_error={facts.get('max_final_output_abs_error')}",
        ]
        return {
            "facts": simple_facts,
            "agent_analysis": "LNE build artifact와 summary count, 수치 오차를 기준으로 이전 Stage 수정 필요 여부를 판단해야 합니다.",
            "recommended_actions": [
                "compile fatal pattern과 build artifact 생성 여부를 먼저 확인합니다.",
                "summary 개수 및 final output error를 기준으로 numerical validation을 재검토합니다.",
            ],
            "needs_stage2_rebuild": bool(facts.get("unsupported_operators")),
        }

    def run_stage3_pipeline(
        self,
        config_mapping: dict[str, Any],
        local_inputs: dict[str, str],
        source_code: str,
        output_dir: str,
        progress_callback: Callable[[str, str], None] | None = None,
        phase: str = "full",
    ) -> dict[str, Any]:
        if progress_callback:
            progress_callback("Initializing", "Stage 3 run directory and LNE JSON payload are being prepared.")

        config = Stage3Config.from_mapping(config_mapping)
        run_id, run_dir = self._make_run_dir(output_dir)
        local_check = self._local_input_check(local_inputs, config)

        tflite_name = Path(local_check.get("selected_tflite_path", "model.tflite")).name
        lne_json_payload = self._build_lne_json_payload(config, tflite_name)
        local_lne_json_path = self._write_local_lne_json(run_dir, lne_json_payload)

        base_summary = {
            "status": "INITIALIZED",
            "run_id": run_id,
            "run_dir": run_dir,
            "execution_mode": config.execution_mode,
            "config_used": {
                "ssh_host": config.ssh_host,
                "ssh_port": config.ssh_port,
                "ssh_user": config.ssh_user,
                "remote_ddk_root": config.remote_ddk_root,
                "remote_tflite_dir": config.remote_tflite_dir,
                "remote_raw_input_dir": config.remote_raw_input_dir,
                "remote_output_dir": config.remote_output_dir,
                "remote_lne_json_path": config.remote_lne_json_path,
                "lne_target": config.lne_target,
                "validation_set": config.validation_set,
                "quant_loop": config.quant_loop,
                "compile_timeout": config.compile_timeout,
                "spike_timeout": config.spike_timeout,
                "execution_mode": config.execution_mode,
            },
            "commands": {"compile": config.build_compile_command(), "spike": config.build_spike_command()},
            "local_inputs": local_check,
            "generated_lne_json_path": local_lne_json_path,
            "generated_lne_json": lne_json_payload,
            "phase": phase,
        }

        missing_config = config.missing_required_fields()
        if missing_config or local_check.get("errors"):
            if progress_callback:
                progress_callback("Blocked", "Required config or local input artifacts are missing.")
            base_summary.update({"status": "NOT_CONFIGURED", "missing_config": missing_config, "local_input_errors": local_check.get("errors", []), "local_input_warnings": local_check.get("warnings", [])})
            self._persist_summary(output_dir, run_dir, base_summary)
            return base_summary

        if config.execution_mode == "dry-run":
            if progress_callback:
                progress_callback("Dry run", "Dry-run mode selected; no remote compile is triggered.")
            validation = self.validator.validate({"execution_mode": "dry-run"}, config.validation)
            base_summary.update({
                "status": "DRY_RUN",
                "validation": validation,
                "remote_design": {
                    "windows_local_role": "Stage 1/2 artifacts 생성 및 Stage 3 orchestrator 실행",
                    "linux_vm_role": "LNE DDK compile, SPIKE simulation, log 생성 및 결과 제공",
                    "transport": "OpenSSH ssh/scp",
                },
            })
            self._persist_summary(output_dir, run_dir, base_summary)
            return base_summary

        backend = self._build_backend(config)
        connection = self.test_connection(config_mapping)
        if connection["status"] != "SUCCESS":
            if progress_callback:
                progress_callback("Connection failed", "SSH connectivity to the Linux VM failed or timed out.")
            validation = self.validator.validate({"execution_mode": "remote", "connection_status": connection["status"]}, config.validation)
            base_summary.update({"status": "CONNECTION_FAILED", "connection": connection, "validation": validation})
            self._persist_summary(output_dir, run_dir, base_summary)
            return base_summary

        upload_result = self._upload_inputs(backend, config, run_id, local_check, local_lne_json_path)

        compile_result = BackendCommandResult(command=config.build_compile_command(), exit_code=-1, stdout="", stderr="", error_type="NOT_RUN")
        compile_analysis = {"conversion_status": "NOT_RUN", "build_status": "NOT_RUN", "artifact_count": 0, "artifact_paths": []}
        spike_result = BackendCommandResult(command=config.build_spike_command(), exit_code=-1, stdout="", stderr="", error_type="NOT_RUN")
        summary_paths: list[str] = []
        downloaded_summaries: list[dict[str, Any]] = []
        spike_analysis = {
            "command": config.build_spike_command(),
            "command_exit_code": -1,
            "command_error_type": "NOT_RUN",
            "fatal_matches": [],
            "summary_paths": [],
            "summary_count": 0,
            "non_empty_summary_count": 0,
            "expected_summary_count": config.validation_set,
            "spike_status": "NOT_RUN",
        }

        if phase in {"full", "lne"}:
            if progress_callback:
                progress_callback("Compiling", "LNE compile is running on the remote VM. Waiting for build status.")
            compile_result = self.converter.run(backend, config)
            artifact_paths = self._discover_artifact_paths(backend, config, local_check["model_name"])
            compile_analysis = self.converter.evaluate_result(compile_result, artifact_paths)
            compile_log_path = self._write_log(Path(run_dir) / "lne_compile.log", f"COMMAND\n{compile_result.command}\n\nSTDOUT\n{compile_result.stdout}\n\nSTDERR\n{compile_result.stderr}")
            self._copy_to_compatibility_path(compile_log_path, Path(output_dir) / "lne_compile.log")
        else:
            compile_log_path = self._write_log(Path(run_dir) / "lne_compile.log", "LNE compile skipped for SPIKE-only phase.")
            self._copy_to_compatibility_path(compile_log_path, Path(output_dir) / "lne_compile.log")

        if phase in {"full", "spike"}:
            if phase == "spike":
                if progress_callback:
                    progress_callback("Compiling prerequisite", "SPIKE-only phase is preparing the current compile state on the VM.")
                compile_result = self.converter.run(backend, config)
                artifact_paths = self._discover_artifact_paths(backend, config, local_check["model_name"])
                compile_analysis = self.converter.evaluate_result(compile_result, artifact_paths)
                compile_log_path = self._write_log(Path(run_dir) / "lne_compile.log", f"COMMAND\n{compile_result.command}\n\nSTDOUT\n{compile_result.stdout}\n\nSTDERR\n{compile_result.stderr}")
                self._copy_to_compatibility_path(compile_log_path, Path(output_dir) / "lne_compile.log")

            if compile_analysis.get("build_status") == "SUCCESS":
                if progress_callback:
                    progress_callback("Running SPIKE", "Compile succeeded; now running the remote SPIKE validation pass.")
                spike_result = self.spike_runner.run(backend, config)
                summary_paths = self._discover_summary_paths(backend, config, local_check["model_name"])
                downloaded_summaries = self._download_summary_files(backend, summary_paths, run_dir)
                non_empty_count = sum(1 for item in downloaded_summaries if item.get("size", 0) > 0)
                spike_analysis = self.spike_runner.evaluate_result(spike_result, summary_paths, non_empty_count, config.validation_set)
            else:
                spike_result = BackendCommandResult(command=config.build_spike_command(), exit_code=-1, stdout="", stderr="Compile did not complete successfully; skipping SPIKE execution.", error_type="SKIPPED")
                spike_analysis = {**spike_analysis, "spike_status": "NOT_RUN", "command_error_type": "SKIPPED"}

        spike_log_path = self._write_log(Path(run_dir) / "spike_validation.log", f"COMMAND\n{spike_result.command}\n\nSTDOUT\n{spike_result.stdout}\n\nSTDERR\n{spike_result.stderr}")
        self._copy_to_compatibility_path(spike_log_path, Path(output_dir) / "spike_validation.log")

        parsed = self.parser.parse(compile_result.stdout, compile_result.stderr, spike_result.stdout, spike_result.stderr, summary_files=downloaded_summaries)

        validation = self.validator.validate(
            {
                "execution_mode": config.execution_mode,
                "connection_status": connection["status"],
                "uploads": upload_result["uploads"],
                "compile_analysis": compile_analysis,
                "spike_analysis": spike_analysis,
                "parsed": parsed,
                "expected_summary_count": config.validation_set,
                "local_input_errors": local_check.get("errors", []),
            },
            config.validation,
        )

        ai_analysis = self._analyze_failure(
            source_code,
            {
                "artifact_count": compile_analysis.get("artifact_count", 0),
                "validation_summary_count": parsed.get("validation_summary_count", 0),
                "max_final_output_abs_error": parsed.get("max_final_output_abs_error"),
                "actual_delta": parsed.get("actual_delta"),
                "qpoint": parsed.get("qpoint"),
                "unsupported_operators": parsed.get("unsupported_operators", []),
            },
            parsed.get("compile_log", ""),
            parsed.get("spike_log", ""),
        )

        summary = {
            **base_summary,
            "status": "COMPLETED",
            "phase": phase,
            "connection": connection,
            "uploads": upload_result["uploads"],
            "remote_paths": {
                "remote_tflite_path": upload_result["remote_tflite_path"],
                "remote_lne_json_path": config.remote_lne_json_path,
                "remote_reference_output_path": upload_result["remote_reference_output_path"],
                "summary_paths": summary_paths,
            },
            "compile": {
                "exit_code": compile_result.exit_code,
                "stdout": compile_result.stdout,
                "stderr": compile_result.stderr,
                "log_path": compile_log_path,
                "status": self.converter.classify(compile_result),
            },
            "compile_analysis": compile_analysis,
            "spike": {
                "exit_code": spike_result.exit_code,
                "stdout": spike_result.stdout,
                "stderr": spike_result.stderr,
                "log_path": spike_log_path,
                "status": self.spike_runner.classify(spike_result),
            },
            "spike_analysis": spike_analysis,
            "downloaded_summaries": [{"remote_path": item["remote_path"], "local_path": item["local_path"], "size": item["size"]} for item in downloaded_summaries],
            "parsed": {
                **parsed,
                "actual_delta_formula": "Max ABS.Error / 2^Qpoint",
                "actual_delta_example": "6 / 2^10 = 0.005859375",
                "lne_guide_contract": {
                    "target": "DQC2",
                    "build_option": "-t dqc",
                    "input_shape": [1, 12, 7],
                    "dtype": "FLOAT32",
                    "quant_loop": config.quant_loop,
                    "validation_set": config.validation_set,
                    "logistic": 1,
                    "sw_act": 1,
                    "fixed_point_format": "FIXEDP16",
                    "expected_worst_actual_delta": 0.005859375,
                },
            },
            "validation": validation,
            "ai_analysis": ai_analysis,
            "remote_design": {
                "windows_local_role": "Stage 1/2 artifacts 생성 및 Stage 3 orchestrator 실행",
                "linux_vm_role": "LNE DDK compile, SPIKE simulation, log 생성 및 결과 제공",
                "transport": "OpenSSH ssh/scp",
                "workspace": config.remote_ddk_root,
            },
        }

        if progress_callback:
            progress_callback("Finished", "Remote LNE compile and SPIKE validation are complete; results are being summarized.")
        self._persist_summary(output_dir, run_dir, summary)
        return summary

    def _persist_summary(self, output_dir: str, run_dir: str, summary: dict[str, Any]) -> None:
        run_summary_path = self._write_log(Path(run_dir) / "stage3_summary.json", json.dumps(summary, indent=2, ensure_ascii=False))
        self._copy_to_compatibility_path(run_summary_path, Path(output_dir) / "stage3_summary.json")

    @staticmethod
    def _quote(value: str) -> str:
        import shlex

        return shlex.quote(value)

    @staticmethod
    def _summarize_log(log_text: str, max_lines: int = 20) -> str:
        text = (log_text or "").strip()
        if not text:
            return "(no log output)"

        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        if len(lines) <= max_lines:
            return "\n".join(lines)

        return "\n".join(lines[:max_lines] + [f"... ({len(lines) - max_lines} lines omitted)"])

    @staticmethod
    def _phase_status_label(status: str | None, phase_name: str) -> str:
        if status in {"SUCCESS", "PASS"}:
            return f"{phase_name} SUCCESS"
        if status in {"FAIL", "ERROR"}:
            return f"{phase_name} FAIL"
        if status in {"WARNING", "NOT_RUN", "NOT_CONFIGURED"}:
            return f"{phase_name} WARNING"
        return f"{phase_name} {status or 'UNKNOWN'}"
