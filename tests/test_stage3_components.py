import unittest

from pipeline_stages.stage3_lne_convert.backend.base import BackendCommandResult
from pipeline_stages.stage3_lne_convert.backend.ssh_linux_vm import SSHLinuxVMBackend
from pipeline_stages.stage3_lne_convert.config import ValidationCriteria
from pipeline_stages.stage3_lne_convert.converter import Stage3Converter
from pipeline_stages.stage3_lne_convert.result_parser import Stage3ResultParser
from pipeline_stages.stage3_lne_convert.spike_runner import Stage3SpikeRunner
from pipeline_stages.stage3_lne_convert.validator import Stage3Validator


class Stage3ConverterTests(unittest.TestCase):
    def test_build_error_with_exit_zero_is_fail(self) -> None:
        converter = Stage3Converter()
        result = BackendCommandResult(
            command="compile",
            exit_code=0,
            stdout="Build Error: partition generation failed",
            stderr="",
        )

        evaluation = converter.evaluate_result(result, ["/remote/output/lne_model_demo"])

        self.assertEqual(evaluation["conversion_status"], "FAIL")
        self.assertEqual(evaluation["build_status"], "FAIL")
        self.assertTrue(evaluation["fatal_matches"])

    def test_git_warning_only_with_artifact_is_success(self) -> None:
        converter = Stage3Converter()
        result = BackendCommandResult(
            command="compile",
            exit_code=0,
            stdout="compile completed",
            stderr="fatal: not a git repository",
        )

        evaluation = converter.evaluate_result(result, ["/remote/output/lne_model_demo"])

        self.assertEqual(evaluation["conversion_status"], "SUCCESS")
        self.assertEqual(evaluation["build_status"], "SUCCESS")
        self.assertIn(r"fatal:\s*not a git repository", evaluation["warning_matches"])
        self.assertEqual(evaluation["fatal_matches"], [])


class Stage3SpikeRunnerTests(unittest.TestCase):
    def test_summary_count_mismatch_is_fail(self) -> None:
        runner = Stage3SpikeRunner()
        result = BackendCommandResult(command="spike", exit_code=0, stdout="PASS", stderr="")

        evaluation = runner.evaluate_result(
            result=result,
            summary_paths=["/remote/output/0001/summary.txt"],
            non_empty_count=1,
            expected_count=2,
        )

        self.assertEqual(evaluation["spike_status"], "FAIL")


class Stage3ResultParserTests(unittest.TestCase):
    def test_last_max_abs_error_is_treated_as_final_output(self) -> None:
        parser = Stage3ResultParser()
        summary_files = [
            {
                "remote_path": "/remote/output/0001/summary.txt",
                "content": "MAX ABS.Error: 0.25\nMAX ABS.Error: 0.50\nMAX ABS.Error: 0.10\n",
            }
        ]

        parsed = parser.parse("", "", "", "", summary_files=summary_files)

        self.assertEqual(parsed["validation_summary_count"], 1)
        self.assertAlmostEqual(parsed["max_intermediate_abs_error"], 0.50)
        self.assertAlmostEqual(parsed["max_final_output_abs_error"], 0.10)
        self.assertEqual(parsed["fallback_rule"], "last_max_abs_error")


class Stage3ValidatorTests(unittest.TestCase):
    def test_threshold_missing_is_not_configured_warning(self) -> None:
        validator = Stage3Validator()
        result = validator.validate(
            {
                "execution_mode": "remote",
                "connection_status": "SUCCESS",
                "uploads": [{"exit_code": 0, "error_type": None}],
                "compile_analysis": {"conversion_status": "SUCCESS", "build_status": "SUCCESS", "fatal_matches": [], "warning_matches": []},
                "spike_analysis": {"spike_status": "SUCCESS", "fatal_matches": []},
                "parsed": {"validation_summary_count": 1, "max_final_output_abs_error": 0.123},
                "expected_summary_count": 1,
            },
            ValidationCriteria(max_abs_error=None, allow_cpu_fallback=False),
        )

        self.assertEqual(result["numerical_validation_status"], "NOT_CONFIGURED")
        self.assertEqual(result["overall_status"], "WARNING")
        self.assertIn("MAX ABS.Error threshold is not configured", result["warnings"])

    def test_dry_run_is_not_pass(self) -> None:
        validator = Stage3Validator()
        result = validator.validate({"execution_mode": "dry-run"}, ValidationCriteria())

        self.assertEqual(result["verdict"], "DRY_RUN")
        self.assertNotEqual(result["verdict"], "PASS")

    def test_worker_unreachable_is_fail(self) -> None:
        validator = Stage3Validator()
        result = validator.validate(
            {
                "execution_mode": "remote",
                "connection_status": "WORKER_UNREACHABLE",
                "uploads": [],
                "compile_analysis": {"conversion_status": "NOT_RUN", "build_status": "NOT_RUN", "fatal_matches": [], "warning_matches": []},
                "spike_analysis": {"spike_status": "NOT_RUN", "fatal_matches": []},
                "parsed": {"validation_summary_count": 0, "max_final_output_abs_error": None},
                "expected_summary_count": 0,
            },
            ValidationCriteria(),
        )

        self.assertEqual(result["overall_status"], "FAIL")
        self.assertIn("Connection failed: WORKER_UNREACHABLE", result["failures"])

    def test_auth_failure_is_fail(self) -> None:
        validator = Stage3Validator()
        result = validator.validate(
            {
                "execution_mode": "remote",
                "connection_status": "SSH_AUTH_FAILED",
                "uploads": [],
                "compile_analysis": {"conversion_status": "NOT_RUN", "build_status": "NOT_RUN", "fatal_matches": [], "warning_matches": []},
                "spike_analysis": {"spike_status": "NOT_RUN", "fatal_matches": []},
                "parsed": {"validation_summary_count": 0, "max_final_output_abs_error": None},
                "expected_summary_count": 0,
            },
            ValidationCriteria(),
        )

        self.assertEqual(result["overall_status"], "FAIL")
        self.assertIn("Connection failed: SSH_AUTH_FAILED", result["failures"])


class SSHLinuxVMBackendTests(unittest.TestCase):
    def test_infer_unreachable_error(self) -> None:
        error_type = SSHLinuxVMBackend.infer_error_type(255, "ssh: connect to host demo port 22: Connection timed out")
        self.assertEqual(error_type, "WORKER_UNREACHABLE")

    def test_infer_auth_failure(self) -> None:
        error_type = SSHLinuxVMBackend.infer_error_type(255, "Permission denied (publickey,password)")
        self.assertEqual(error_type, "SSH_AUTH_FAILED")


if __name__ == "__main__":
    unittest.main()
