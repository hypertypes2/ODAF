from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline_stages.stage1_model_design.agent import ModelDesignAgent
from pipeline_stages.stage1_model_design.data_prep import DataPreparationAgent
from session import ODAFSessionState


def render_stage1() -> None:
    st.header("1️⃣ Stage 1: Model Design")
    st.caption("데이터 준비, 모델 설계, 코드 분석 및 최소 수정까지 하나의 agentic stage로 수행합니다.")

    tabs = st.tabs(["📊 Data Prep", "🧠 Model Design", "📁 Artifacts"])
    default_code = st.session_state.get(
        "stage1_original_code",
        "model = tf.keras.Sequential()\nmodel.add(layers.GRU(64, return_sequences=True))\nmodel.add(layers.Dense(5))",
    )

    with tabs[0]:
        _render_data_prep_tab()

    with tabs[1]:
        source_code_input = st.text_area(
            "💻 훈련/설계용 파이썬 소스코드 입력",
            value=default_code,
            height=220,
            key="s1_raw_code_area",
        )

        col1, col2 = st.columns(2)
        with col1:
            seq_len = st.number_input("Sequence Length", min_value=1, value=int(st.session_state.get("parsed_seq_len", 24)))
        with col2:
            d_model = st.number_input("Input Feature Count", min_value=1, value=int(st.session_state.get("parsed_d_model", 5)))
        st.session_state.parsed_seq_len = int(seq_len)
        st.session_state.parsed_d_model = int(d_model)

        if st.button("🔧 AI 에이전트 분석 및 최소 리팩터링", type="primary", key="s1_trigger_btn"):
            with st.spinner("Stage 1 AI Agent가 모델 목적과 제약조건을 분석 중입니다..."):
                agent = ModelDesignAgent()
                result = agent.run_stage1_pipeline(source_code_input)
                st.session_state.s1_pipeline_data = result
                st.session_state.stage1_original_code = result["original_code"]
                st.session_state.stage1_refactored_code = result["final_code"]
                st.session_state.stage1_code_diff = result["change_report"]["diff"]
                _save_stage1_code_artifacts(result["original_code"], result["final_code"])
                ODAFSessionState.reset_downstream(1)

        result = st.session_state.get("s1_pipeline_data")
        if result:
            _render_model_analysis(result)

            if st.button("✅ Stage 1 승인 및 Stage 2로 전달", key="s1_complete_btn"):
                summary = {
                    "data_prep": st.session_state.get("stage1_data_prep_summary", {}),
                    "model_name": result["analysis"].get("model_name", "Unknown"),
                    "model_type": result["analysis"].get("base_model_type", "Unknown"),
                    "layer_config": ", ".join(result["analysis"].get("parsed_operators", [])),
                    "total_params": "N/A",
                    "lne_compliant": "준수" if result["simulation"]["status"] == "SUCCESS" else "보정 필요",
                    "modification_reason": result["change_report"]["modification_reason"],
                    "code_artifacts": st.session_state.get("stage1_code_artifacts", {}),
                }
                st.session_state.stage1_summary = summary
                ODAFSessionState.mark_stage_complete(1, summary)
                st.success("🎉 Stage 1 완료! 사이드바에서 Stage 2로 이동하세요.")

    with tabs[2]:
        artifacts = st.session_state.get("stage1_code_artifacts", {})
        if artifacts:
            st.json(artifacts)
        else:
            st.info("아직 생성된 Stage 1 아티팩트가 없습니다.")

        diff_text = st.session_state.get("stage1_code_diff", "")
        if diff_text:
            st.code(diff_text, language="diff")


def _render_data_prep_tab() -> None:
    st.subheader("Stage 1 데이터 준비")
    st.caption("기존 Stage 0 기능을 Stage 1 내부로 이동해 재사용합니다.")

    uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"], key="stage1_csv_uploader")
    seq_len = st.number_input("기본 시퀀스 길이", min_value=1, value=int(st.session_state.get("parsed_seq_len", 12)), key="stage1_seq_len_input")
    st.session_state.parsed_seq_len = int(seq_len)

    if uploaded_file is None:
        existing_path = st.session_state.get("processed_data_path", "")
        if existing_path:
            st.success(f"기존 전처리 데이터 사용 가능: {existing_path}")
        else:
            st.info("CSV를 업로드하면 Stage 1에서 바로 분석/전처리할 수 있습니다.")
        return

    prep_agent = DataPreparationAgent()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = Path(temp_file.name)

    try:
        dataframe, stats = prep_agent.load_and_analyze(temp_path)
        st.success(f"✅ 데이터 로드 완료: {stats['total_rows']}행 × {stats['total_columns']}열")
        st.dataframe(dataframe.head(10), width="stretch")

        remove_duplicates = st.checkbox("중복 제거", value=True, key="stage1_remove_duplicates")
        remove_na = st.checkbox("결측치 제거", value=True, key="stage1_remove_na")
        normalize = st.checkbox("Min-Max 정규화", value=True, key="stage1_normalize")

        for image_name, image_bytes in prep_agent.generate_visualizations(dataframe).items():
            st.image(image_bytes, caption=image_name, width="stretch")

        if st.button("🔄 Stage 1 데이터 전처리 실행", key="stage1_preprocess_btn"):
            processed = prep_agent.preprocess_data(
                dataframe,
                remove_duplicates=remove_duplicates,
                remove_na=remove_na,
                normalize=normalize,
            )
            output_path = ODAFSessionState.artifact_path("stage1_model_design", "processed_data.csv")
            saved_path = prep_agent.save_processed_data(processed, output_path)
            st.session_state.processed_data_path = saved_path
            st.session_state.parsed_d_model = int(min(len(processed.columns), max(1, st.session_state.get("parsed_d_model", 5))))
            summary = {
                "total_rows": stats["total_rows"],
                "num_features": stats["total_columns"],
                "missing_handled": "완료" if remove_na else "미적용",
                "normalization": "Min-Max Scaling" if normalize else "미적용",
                "processed_data_path": saved_path,
                "preprocessing_log": prep_agent.preprocessing_log,
            }
            st.session_state.stage1_data_prep_summary = summary
            st.success("전처리 데이터를 Stage 1 출력 경로에 저장했습니다.")
            st.code(prep_agent.get_preprocessing_summary())
            st.dataframe(processed.head(10), width="stretch")
    finally:
        temp_path.unlink(missing_ok=True)


def _render_model_analysis(result: dict) -> None:
    st.subheader("🔍 모델 분석 결과")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("모델명", result["analysis"].get("model_name", "Unknown"))
        st.metric("모델 구조", result["analysis"].get("base_model_type", "Unknown"))
    with col2:
        st.metric("원본 가중치 추정", f"{result['analysis'].get('raw_weight_mb', 0.0)} MB")
        unroll_status = "확보" if result["analysis"].get("has_unroll_true", False) else "누락"
        st.metric("RNN Unroll", unroll_status)

    st.markdown("**검출된 Operators**")
    st.code(str(result["analysis"].get("parsed_operators", [])))

    if result["simulation"]["status"] == "FAILED":
        st.error(result["simulation"]["error_message"])
    else:
        st.success("LNE 제약 시뮬레이션을 통과했습니다.")

    st.markdown("### 변경 전후")
    before_column, after_column = st.columns(2)
    with before_column:
        st.caption("원본 코드")
        st.code(result["original_code"], language="python")
    with after_column:
        st.caption("리팩터링 코드")
        st.code(result["final_code"], language="python")

    change_report = result["change_report"]
    st.info(change_report["modification_reason"])
    if change_report["diff"]:
        st.code(change_report["diff"], language="diff")

    st.markdown("### 수학적 동등성")
    st.markdown(result["mathematical_equivalence"])


def _save_stage1_code_artifacts(original_code: str, refactored_code: str) -> None:
    original_path = ODAFSessionState.artifact_path("stage1_model_design", "original_model.py")
    refactored_path = ODAFSessionState.artifact_path("stage1_model_design", "refactored_model.py")
    Path(original_path).write_text(original_code, encoding="utf-8")
    Path(refactored_path).write_text(refactored_code, encoding="utf-8")
    st.session_state.stage1_code_artifacts = {
        "original_code_path": str(original_path),
        "refactored_code_path": str(refactored_path),
        "processed_data_path": st.session_state.get("processed_data_path", ""),
    }
