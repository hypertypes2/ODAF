from __future__ import annotations

from .config import ValidationCriteria


class Stage3Validator:
    def validate(self, stage_data: dict, criteria: ValidationCriteria) -> dict:
        failures: list[str] = []
        warnings: list[str] = []

        execution_mode = stage_data.get("execution_mode", "remote")
        if execution_mode == "dry-run":
            return {
                "connection_status": "DRY_RUN",
                "upload_status": "DRY_RUN",
                "conversion_status": "DRY_RUN",
                "build_status": "DRY_RUN",
                "spike_status": "DRY_RUN",
                "validation_count_status": "DRY_RUN",
                "numerical_validation_status": "DRY_RUN",
                "overall_status": "DRY_RUN",
                "verdict": "DRY_RUN",
                "failures": [],
                "warnings": ["Dry-run does not execute remote compile or SPIKE."],
            }

        parsed = stage_data.get("parsed", {})
        connection_status = stage_data.get("connection_status", "NOT_RUN")

        uploads = stage_data.get("uploads", [])
        if uploads and all(item.get("exit_code") == 0 and not item.get("error_type") for item in uploads):
            upload_status = "SUCCESS"
        elif stage_data.get("local_input_errors"):
            upload_status = "FAIL"
            failures.extend(stage_data.get("local_input_errors", []))
        else:
            upload_status = "FAIL" if uploads else "NOT_RUN"

        compile_analysis = stage_data.get("compile_analysis", {})
        conversion_status = compile_analysis.get("conversion_status", "NOT_RUN")
        build_status = compile_analysis.get("build_status", "NOT_RUN")
        if compile_analysis.get("fatal_matches"):
            failures.append("LNE compile fatal pattern detected")
        if compile_analysis.get("warning_matches"):
            warnings.extend(compile_analysis.get("warning_matches", []))

        spike_analysis = stage_data.get("spike_analysis", {})
        spike_status = spike_analysis.get("spike_status", "NOT_RUN")
        if spike_analysis.get("fatal_matches"):
            failures.append("SPIKE fatal pattern detected")

        expected_summary_count = stage_data.get("expected_summary_count", 0)
        actual_summary_count = parsed.get("validation_summary_count", 0)
        if expected_summary_count and actual_summary_count >= expected_summary_count:
            validation_count_status = "SUCCESS"
        elif expected_summary_count:
            validation_count_status = "FAIL"
            failures.append(f"Expected at least {expected_summary_count} summary.txt files but found {actual_summary_count}")
        else:
            validation_count_status = "NOT_CONFIGURED"
            warnings.append("Validation summary count is not configured")

        max_abs_error = parsed.get("max_final_output_abs_error")
        if max_abs_error is None:
            max_abs_error = parsed.get("max_abs_error")
        qpoint = parsed.get("qpoint") or criteria.qpoint
        actual_delta = parsed.get("actual_delta")
        numerical_validation_status = "NOT_CONFIGURED"
        if actual_delta is None and max_abs_error is not None and qpoint:
            actual_delta = float(max_abs_error) / float(2 ** int(qpoint))

        if actual_delta is not None:
            if criteria.expected_worst_delta is None:
                warnings.append("Worst actual delta threshold is not configured")
                numerical_validation_status = "NOT_CONFIGURED"
            else:
                nominal_limit = float(criteria.expected_worst_delta)
                tolerance_limit = nominal_limit * 2.0
                if actual_delta <= nominal_limit:
                    numerical_validation_status = "SUCCESS"
                elif actual_delta <= tolerance_limit:
                    warnings.append(
                        f"Actual delta {actual_delta} is above the nominal threshold {nominal_limit}, but within the accepted tolerance window {tolerance_limit}; treating as PASS."
                    )
                    numerical_validation_status = "SUCCESS"
                else:
                    numerical_validation_status = "FAIL"
                    failures.append(f"Actual delta {actual_delta} exceeded threshold {criteria.expected_worst_delta}")
        else:
            numerical_validation_status = "WARNING"
            warnings.append("No final output MAX ABS.Error or Q-point was parsed")

        if parsed.get("lne_contract_checks"):
            checks = parsed["lne_contract_checks"]
            for name, passed in checks.items():
                if not passed:
                    warnings.append(f"LNE contract check failed: {name}")

        if parsed.get("cpu_fallback_detected"):
            if criteria.allow_cpu_fallback is None:
                warnings.append("CPU fallback policy is not configured")
            elif criteria.allow_cpu_fallback is False:
                failures.append("CPU fallback detected but policy forbids it")

        if parsed.get("unsupported_operators"):
            warnings.append("Unsupported operator detected")
        if parsed.get("compile_errors"):
            warnings.append("Compile error text detected in logs")

        if connection_status in {"SSH_AUTH_FAILED", "WORKER_UNREACHABLE", "FAIL"}:
            failures.append(f"Connection failed: {connection_status}")

        overall_status = "PASS"
        fail_like = {"FAIL", "SSH_AUTH_FAILED", "WORKER_UNREACHABLE", "COMMAND_TIMEOUT", "SSH_BINARY_NOT_FOUND"}
        statuses = [connection_status, upload_status, conversion_status, build_status, spike_status, validation_count_status, numerical_validation_status]
        if any(status in fail_like for status in statuses):
            overall_status = "FAIL"
        elif (
            build_status == "SUCCESS"
            and spike_status == "SUCCESS"
            and validation_count_status == "SUCCESS"
            and numerical_validation_status == "SUCCESS"
            and not failures
        ):
            overall_status = "PASS"
        elif any(status in {"WARNING", "NOT_CONFIGURED", "NOT_RUN"} for status in statuses) or warnings:
            overall_status = "WARNING"
        elif failures:
            overall_status = "FAIL"

        return {
            "connection_status": connection_status,
            "upload_status": upload_status,
            "conversion_status": conversion_status,
            "build_status": build_status,
            "spike_status": spike_status,
            "validation_count_status": validation_count_status,
            "numerical_validation_status": numerical_validation_status,
            "overall_status": overall_status,
            "verdict": overall_status,
            "failures": failures,
            "warnings": warnings,
        }
