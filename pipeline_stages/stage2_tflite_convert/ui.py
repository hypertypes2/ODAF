import os
import streamlit as st
import pandas as pd
from pipeline_stages.stage2_tflite_convert.agent import DatasetDrivenTFLiteConvertAgent
from session import ODAFSessionState

def render_stage2_interface():
    st.header("2️⃣ Stage 2: TFLite 변환 및 캘리브레이션 데이터셋 생성")
    st.caption("SavedModel, TFLite, Calibration, LNE 입력 산출물과 변환 실패 분석을 포함한 agentic convert stage입니다.")
    
    seq_len = st.session_state.get("parsed_seq_len", 24)
    d_model = st.session_state.get("parsed_d_model", 5)
    stage1_code = st.session_state.get("stage1_refactored_code", "") or st.session_state.get("stage1_original_code", "")
    processed_data_path = st.session_state.get("processed_data_path", "")
    csv_for_conversion = ""

    st.subheader("⚙️ 입력 데이터 설정")
    
    # 1. Stage 1 출력 소스코드 표시
    st.markdown("#### 📄 Stage 1 출력 소스코드")
    if stage1_code:
        with st.expander("소스코드 미리보기", expanded=False):
            st.code(stage1_code, language='python')
    else:
        st.warning("⚠️ Stage 1을 먼저 완료하여 최적화된 소스코드를 생성하세요.")
    
    st.markdown("#### 📊 학습/변환 데이터셋")
    csv_candidate_path = st.text_input(
        "변환용 CSV 경로",
        value=processed_data_path or "",
        key="stage2_csv_path_input",
        help="예: Stage 1 전처리 CSV 또는 calibration_reference.csv 경로",
    ).strip()

    if csv_candidate_path and os.path.exists(csv_candidate_path):
        csv_for_conversion = csv_candidate_path
        st.success(f"📁 변환 입력 데이터 사용: {csv_for_conversion}")
        try:
            preview_df = pd.read_csv(csv_for_conversion)
            st.caption(f"💡 총 행 수: {len(preview_df)}개 | 총 컬럼 수: {len(preview_df.columns)}개")
            
            if not all(
                column in preview_df.columns for column in ["T_in", "T_sp", "Hz", "T_out", "RH", "Fan", "w_out"]
            ):
                st.caption("💡 입력 채널은 실제 온도 모델 입력 규격과 일치하도록 기본 7개 특성으로 계산됩니다.")
        
        except Exception as e:
            st.error(f"❌ CSV 읽기 실패: {e}")
    else:
        st.warning("⚠️ 유효한 CSV 경로가 필요합니다.")
    
    # 3. 타겟 보드 선택
    target_board = st.selectbox(
        "🎛️ 타겟 하드웨어 칩셋 선택", 
        ["DQC2 (고성능 칩셋)", "DQC1 (저전력 칩셋)"], 
        key="s2_board"
    )
    
    # 4. TFLite 변환 실행
    if st.button("🚀 Float32 TFLite 변환 및 캘리브레이션 데이터 생성", type="primary", key="s2_convert_btn"):
        if not stage1_code:
            st.error("❌ Stage 1 소스코드가 없습니다. Stage 1을 먼저 완료하세요.")
        elif not (csv_for_conversion and os.path.exists(csv_for_conversion)):
            st.error("❌ 유효한 CSV 경로가 필요합니다.")
        else:
            agent = DatasetDrivenTFLiteConvertAgent(seq_len=seq_len, d_model=d_model)
            
            with st.spinner("TFLite 변환 및 캘리브레이션 데이터 생성 중..."):
                try:
                    output_dir = str(ODAFSessionState.stage_output_dir("stage2_tflite_convert"))
                    result = agent.run_stage2_pipeline(
                        source_code=stage1_code,
                        csv_path=csv_for_conversion,
                        target_board=target_board,
                        output_dir=output_dir
                    )

                    if not result.tflite_path:
                        st.error("❌ TFLite 변환 실패")
                        st.json(result.failure_analysis)
                        if os.path.exists(result.conversion_log_path):
                            with open(result.conversion_log_path, 'r', encoding='utf-8') as log_file:
                                st.code(log_file.read())

                        if result.suggested_code_patch:
                            st.subheader("🛠️ Stage 2 제안 최소 수정 코드")
                            st.code(result.suggested_code_patch, language="python")
                            if st.button("이 수정안을 Stage 1 코드로 반영", key="stage2_apply_patch"):
                                st.session_state.stage1_refactored_code = result.suggested_code_patch
                                ODAFSessionState.reset_downstream(1)
                                st.success("Stage 1 코드에 수정안을 반영했습니다. 다시 Stage 2를 실행하세요.")
                        return

                    st.success("✅ TFLite 변환 완료!")
                    
                    # 결과 표시
                    st.subheader("📊 변환 결과")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("FP32 모델 크기", f"{result.model_size_kb} KB")
                    with col2:
                        st.metric("Calibration 샘플 수", f"{result.num_calibration_samples}")
                    with col3:
                        st.metric("변환 모드", "FP32 Only")
                    
                    # 정확도 비교
                    st.info("현재 Stage 2는 LNE DDK 후속 양자화를 위해 Float32 TFLite만 생성합니다.")
                    
                    # 벤치마킹
                    st.markdown("---")
                    st.subheader("🚀 하드웨어 벤치마크 예측")
                    bench_result = agent.run_benchmarking(target_board, result.model_size_kb)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("예상 추론 속도 (INT8)", f"{bench_result['estimated_latency_ms']} ms")
                    with col2:
                        st.metric("타겟 칩셋", target_board)
                    
                    if "WARNING" in bench_result['status']:
                        st.warning(f"⚠️ {bench_result['status']}")
                    else:
                        st.success(f"✅ {bench_result['status']}")
                    
                    # 다운로드 패키지 생성
                    st.markdown("---")
                    st.subheader("📦 배포 패키지 다운로드")
                    
                    zip_path = agent.create_download_package(result, output_zip=os.path.join(output_dir, "odaf_package.zip"))
                    
                    with open(zip_path, 'rb') as f:
                        st.download_button(
                            label="⬇️ 전체 패키지 다운로드 (ZIP)",
                            data=f.read(),
                            file_name="odaf_deployment_package.zip",
                            mime="application/zip",
                            type="primary"
                        )
                    
                    st.caption("📁 포함 내용: FP32 모델, 캘리브레이션 데이터셋, LNE 입력/레퍼런스")
                    
                    # 개별 파일 다운로드
                    with st.expander("개별 파일 다운로드"):
                        with open(result.tflite_path, 'rb') as f:
                            st.download_button(
                                label="📄 FP32 모델 다운로드",
                                data=f.read(),
                                file_name="model_fp32.tflite",
                                mime="application/octet-stream"
                            )

                        if result.tflite_int8_path and os.path.exists(result.tflite_int8_path):
                            with open(result.tflite_int8_path, 'rb') as f:
                                st.download_button(
                                    label="📄 INT8 모델 다운로드",
                                    data=f.read(),
                                    file_name="model_int8.tflite",
                                    mime="application/octet-stream"
                                )
                    
                    # 세션에 저장
                    st.session_state.stage2_tflite_path = result.tflite_path
                    st.session_state.stage2_fp32_tflite_path = result.tflite_path
                    st.session_state.stage2_saved_model_dir = result.saved_model_dir
                    st.session_state.stage2_calibration_folder = result.calibration_folder
                    st.session_state.stage2_raw_input_dir = result.raw_input_dir
                    st.session_state.stage2_reference_output_path = result.reference_output_path
                    st.session_state.stage2_conversion_log_path = result.conversion_log_path
                    st.session_state.stage2_retry_history_path = result.retry_history_path
                    st.session_state.stage2_done = True
                    
                    st.session_state.stage2_conversion_summary = {
                        "fp32_size_kb": result.model_size_kb,
                        "int8_size_kb": result.model_int8_size_kb,
                        "compression_ratio": 0,
                        "fp32_accuracy": result.fp32_accuracy,
                        "int8_accuracy": result.int8_accuracy,
                        "accuracy_delta": abs(result.fp32_accuracy - result.int8_accuracy),
                        "saved_model_dir": result.saved_model_dir,
                        "raw_input_dir": result.raw_input_dir,
                        "reference_output_path": result.reference_output_path,
                        "conversion_log_path": result.conversion_log_path,
                        "retry_history_path": result.retry_history_path,
                        "lne_contract": {
                            "target": "DQC2",
                            "build_option": "-t dqc",
                            "input_shape": [1, 12, 7],
                            "dtype": "FLOAT32",
                            "raw_input_count": 200,
                            "raw_input_file_size_bytes": 336,
                            "quant_loop": 10,
                            "validation_set": 10,
                            "logistic": 1,
                            "sw_act": 1,
                            "max_lne_partitions": 2,
                            "fixed_point_format": "FIXEDP16",
                            "output_qpoint": 10,
                            "expected_worst_actual_delta": 0.005859375,
                        },
                        "actual_delta_rule": "Real Delta = Max ABS.Error / 2^Qpoint; worst actual delta = 6 / 2^10 = 0.005859375",
                    }
                    st.session_state.stage2_data = st.session_state.stage2_conversion_summary
                    
                    st.markdown("---")
                    st.write("SavedModel, FP32/INT8 TFLite, Calibration Data, LNE Raw Input, Reference Output이 모두 준비되었습니다.")
                    if st.checkbox("✅ Stage 2 완료 - Stage 3로 진행", key="stage2_complete_check"):
                        st.success("🎉 Stage 2 완료! 사이드바에서 Stage 3로 이동하세요.")
                    
                except Exception as e:
                    st.error(f"❌ 변환 실패: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    else:
        st.info("💡 Stage 1 소스코드와 전처리 데이터를 준비한 후 버튼을 눌러주세요.")
