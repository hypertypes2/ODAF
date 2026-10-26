from __future__ import annotations

import re
from typing import Any


class Stage3ResultParser:
    MAX_ABS_PATTERN = re.compile(r"MAX\s*ABS\.?\s*Error\s*[:=]\s*([0-9eE+\-.]+)", flags=re.IGNORECASE)
    NAME_PATTERN = re.compile(r"(?:tensor|layer|output)\s*[:=]\s*([A-Za-z0-9_./-]+)", flags=re.IGNORECASE)
    QPOINT_PATTERN = re.compile(r"DataQ\s*[:=]\s*(\d+)|Qpoint\s*[:=]\s*(\d+)|Q-Point\s*[:=]\s*(\d+)", flags=re.IGNORECASE)
    DELTA_PATTERN = re.compile(r"Real\s*Delta\s*[:=]\s*([0-9eE+\-.]+)|actual\s*delta\s*[:=]\s*([0-9eE+\-.]+)", flags=re.IGNORECASE)

    def parse(
        self,
        compile_stdout: str,
        compile_stderr: str,
        spike_stdout: str,
        spike_stderr: str,
        summary_files: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        compile_log = "\n".join([compile_stdout or "", compile_stderr or ""])
        spike_log = "\n".join([spike_stdout or "", spike_stderr or ""])
        combined_log = "\n".join([compile_log, spike_log])

        mapper_flags = re.findall(r"Mapper\s*flag\s*[:=]\s*(.+)", combined_log, flags=re.IGNORECASE)
        custom_ops = re.findall(r"LNE_CUSTOM_OP\s*[:=]?\s*([A-Za-z0-9_./-]+)", combined_log, flags=re.IGNORECASE)
        unsupported_operators = re.findall(r"unsupported\s+operator\s*[:=]?\s*([A-Za-z0-9_./-]+)", combined_log, flags=re.IGNORECASE)
        cpu_mapping = re.findall(r"CPU\s*Mapping\s*[:=]\s*(.+)", combined_log, flags=re.IGNORECASE)
        lne_mapping = re.findall(r"LNE\s*Mapping\s*[:=]\s*(.+)", combined_log, flags=re.IGNORECASE)
        warnings = re.findall(r"warning[:\]]\s*(.+)", combined_log, flags=re.IGNORECASE)
        errors = re.findall(r"error[:\]]\s*(.+)", combined_log, flags=re.IGNORECASE)
        max_abs_match = re.search(r"MAX\s*ABS\.?\s*Error\s*[:=]\s*([0-9eE+\-.]+)", combined_log, flags=re.IGNORECASE)
        sample_results = re.findall(r"sample\s*(\d+)\s*[:=]\s*(PASS|FAIL)", combined_log, flags=re.IGNORECASE)

        qpoint_value = None
        qpoint_match = self.QPOINT_PATTERN.search(combined_log)
        if qpoint_match:
            for group in qpoint_match.groups():
                if group is not None:
                    qpoint_value = int(group)
                    break

        delta_float = None
        delta_match = self.DELTA_PATTERN.search(combined_log)
        if delta_match:
            for group in delta_match.groups():
                if group is not None:
                    delta_float = float(group)
                    break

        summary_payload = self.parse_summary_files(summary_files or [])

        result = {
            "compile_log": compile_log,
            "spike_log": spike_log,
            "mapper_flags": mapper_flags,
            "custom_ops": custom_ops,
            "cpu_fallback_detected": "cpu fallback" in combined_log.lower(),
            "cpu_mapping": cpu_mapping,
            "lne_mapping": lne_mapping,
            "unsupported_operators": unsupported_operators,
            "compile_warnings": warnings,
            "compile_errors": errors,
            "spike_success": "pass" in spike_log.lower() or "success" in spike_log.lower(),
            "max_abs_error": float(max_abs_match.group(1)) if max_abs_match else None,
            "sample_results": sample_results,
            "qpoint": qpoint_value,
            "actual_delta": delta_float,
            "lne_contract_checks": {
                "float32_input": "FLOAT32" in combined_log.upper() or "[1, 12, 7]" in combined_log or "1, 12, 7" in combined_log,
                "fixedp16": "FIXEDP16" in combined_log.upper(),
                "logistic_1": "logistic=1" in combined_log.lower() or '"logistic": 1' in combined_log.lower() or "logistic_1" in combined_log.lower(),
                "sw_act_1": "sw_act=1" in combined_log.lower() or '"sw_act": 1' in combined_log.lower() or "sw_act_1" in combined_log.lower(),
                "single_lne_custom_op": "LNE_CUSTOM_OP" in combined_log.upper() and "TANH" not in combined_log.upper(),
            },
        }
        result.update(summary_payload)
        if result.get("max_final_output_abs_error") is not None:
            result["max_abs_error"] = result["max_final_output_abs_error"]

        summary_max_error = result.get("max_final_output_abs_error")
        if summary_max_error is not None and qpoint_value:
            result["actual_delta"] = float(summary_max_error) / float(2 ** qpoint_value)

        return result

    def parse_summary_files(self, summary_files: list[dict[str, Any]]) -> dict[str, Any]:
        samples: list[dict[str, Any]] = []
        final_errors: list[float] = []
        intermediate_errors: list[float] = []
        fallback_used = False

        for index, summary in enumerate(summary_files):
            content = summary.get("content", "")
            remote_path = summary.get("remote_path", "")
            all_errors = [float(match) for match in self.MAX_ABS_PATTERN.findall(content)]
            named_errors = self._extract_named_errors(content)
            sample_id = self._extract_sample_id(remote_path or summary.get("local_path", ""), index)

            final_output_error = None
            intermediate_max = None
            parser_confidence = "fallback"
            fallback_rule = None

            explicit_output_values = [entry["value"] for entry in named_errors if "output" in (entry["name"] or "").lower()]
            if explicit_output_values:
                final_output_error = explicit_output_values[-1]
                remaining_values = [entry["value"] for entry in named_errors[:-1]] or [value for value in all_errors if value != final_output_error]
                intermediate_max = max(remaining_values) if remaining_values else None
                parser_confidence = "named_output"
            elif all_errors:
                fallback_used = True
                fallback_rule = "last_max_abs_error"
                final_output_error = all_errors[-1]
                intermediate_values = all_errors[:-1]
                intermediate_max = max(intermediate_values) if intermediate_values else None

            if final_output_error is not None:
                final_errors.append(final_output_error)
            if intermediate_max is not None:
                intermediate_errors.append(intermediate_max)

            samples.append(
                {
                    "sample_id": sample_id,
                    "summary_path": remote_path,
                    "all_max_abs_errors": all_errors,
                    "intermediate_max_abs_error": intermediate_max,
                    "final_output_max_abs_error": final_output_error,
                    "parser_confidence": parser_confidence,
                    "fallback_rule": fallback_rule,
                }
            )

        return {
            "validation_summary_count": len(summary_files),
            "samples": samples,
            "max_final_output_abs_error": max(final_errors) if final_errors else None,
            "max_intermediate_abs_error": max(intermediate_errors) if intermediate_errors else None,
            "parser_confidence": "fallback" if fallback_used else ("named_output" if summary_files else "not_available"),
            "fallback_rule": "last_max_abs_error" if fallback_used else None,
        }

    def _extract_named_errors(self, content: str) -> list[dict[str, Any]]:
        named_errors: list[dict[str, Any]] = []
        lines = content.splitlines()
        for line_index, line in enumerate(lines):
            match = self.MAX_ABS_PATTERN.search(line)
            if not match:
                continue
            name_match = self.NAME_PATTERN.search(line)
            if not name_match and line_index > 0:
                name_match = self.NAME_PATTERN.search(lines[line_index - 1])
            named_errors.append(
                {
                    "name": name_match.group(1) if name_match else None,
                    "value": float(match.group(1)),
                }
            )
        return named_errors

    def _extract_sample_id(self, path_hint: str, index: int) -> str:
        match = re.search(r"(\d{4,})", path_hint)
        if match:
            return match.group(1)
        return f"{index:08d}"
