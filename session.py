from pathlib import Path
from typing import Any, Dict, List

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"


class ODAFSessionState:
    """ODAF 파이프라인의 세션 상태, 출력 경로, stage 흐름을 관리합니다."""

    STAGES: List[Dict[str, Any]] = [
        {"index": 1, "key": "stage1", "slug": "stage1_model_design", "title": "Model Design"},
        {"index": 2, "key": "stage2", "slug": "stage2_tflite_convert", "title": "TFLite Convert"},
        {"index": 3, "key": "stage3", "slug": "stage3_lne_convert", "title": "LNE Convert + SPIKE Validation"},
        {"index": 4, "key": "stage4", "slug": "stage4_production_simulation", "title": "Production Simulation"},
        {"index": 5, "key": "stage5", "slug": "stage5_executive_sign", "title": "Executive Sign"},
    ]

    DEFAULTS: Dict[str, Any] = {
        "selected_stage": 1,
        "pipeline_complete": False,
        "parsed_seq_len": 12,
        "parsed_d_model": 7,
        "stage1_done": False,
        "stage2_done": False,
        "stage3_done": False,
        "stage4_done": False,
        "stage5_done": False,
        "stage1_approved": False,
        "stage3_verified": False,
        "stage1_data": None,
        "stage2_data": None,
        "stage3_data": None,
        "stage4_data": None,
        "stage5_data": None,
        "stage1_summary": {},
        "stage2_conversion_summary": {},
        "stage3_validation_summary": {},
        "stage4_simulation_summary": {},
        "stage5_approval_summary": {},
        "stage1_refactored_code": "",
        "stage1_original_code": "",
        "stage1_code_diff": "",
        "stage2_tflite_path": "",
        "stage2_fp32_tflite_path": "",
        "stage2_saved_model_dir": "",
        "stage2_calibration_folder": "",
        "stage2_raw_input_dir": "",
        "stage2_reference_output_path": "",
        "stage2_conversion_log_path": "",
        "stage2_retry_history_path": "",
        "stage3_result": {},
        "stage4_result": {},
        "stage5_result": {},
        "processed_data_path": "",
        "remote_worker_config": {},
    }

    @classmethod
    def initialize(cls) -> None:
        OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
        for stage in cls.STAGES:
            cls.stage_output_dir(stage["slug"])

        for key, value in cls.DEFAULTS.items():
            if key not in st.session_state:
                st.session_state[key] = value.copy() if isinstance(value, dict) else value

    @classmethod
    def stage_definitions(cls) -> List[Dict[str, Any]]:
        return cls.STAGES

    @staticmethod
    def outputs_root() -> Path:
        OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
        return OUTPUTS_ROOT

    @classmethod
    def stage_output_dir(cls, stage_slug: str) -> Path:
        path = cls.outputs_root() / stage_slug
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def artifact_path(cls, stage_slug: str, *parts: str) -> Path:
        path = cls.stage_output_dir(stage_slug)
        for part in parts:
            path = path / part
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def set_selected_stage(stage_index: int) -> None:
        st.session_state.selected_stage = stage_index

    @staticmethod
    def mark_stage_complete(stage_index: int, summary: Dict[str, Any] | None = None) -> None:
        state_key = f"stage{stage_index}_done"
        st.session_state[state_key] = True
        if stage_index == 1:
            st.session_state.stage1_approved = True
        if stage_index == 3:
            st.session_state.stage3_verified = True
        if summary is not None:
            st.session_state[f"stage{stage_index}_data"] = summary

    @staticmethod
    def reset_downstream(from_stage: int) -> None:
        for stage_index in range(from_stage + 1, 6):
            st.session_state[f"stage{stage_index}_done"] = False
            st.session_state[f"stage{stage_index}_data"] = None

        st.session_state.stage3_verified = False
        st.session_state.pipeline_complete = False
        st.session_state.stage2_conversion_summary = {} if from_stage < 2 else st.session_state.stage2_conversion_summary
        st.session_state.stage3_validation_summary = {} if from_stage < 3 else st.session_state.stage3_validation_summary
        st.session_state.stage4_simulation_summary = {} if from_stage < 4 else st.session_state.stage4_simulation_summary
        st.session_state.stage5_approval_summary = {} if from_stage < 5 else st.session_state.stage5_approval_summary

