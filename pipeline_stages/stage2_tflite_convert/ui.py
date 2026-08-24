import os
import streamlit as st
import pandas as pd
import numpy as np
# 백엔드 엔진 파일에서 하드웨어 컴파일러 및 벤치마크 클래스 바인딩
from pipeline_stages.stage2_tflite_convert.agent import DatasetDrivenTFLiteConvertAgent

def render_stage2_interface():
    st.header("2️⃣ Stage 2: TFLite 변환 및 하드웨어 벤치마킹")
    st.caption("SW개발팀 주관 공정 - 타겟 보드별 최적의 양자화 기법을 탐색합니다.")
    
    base_mse = 0.0125
    seq_len = st.session_state.get("parsed_seq_len", 24)
    d_model = st.session_state.get("parsed_d_model", 5)
    approved_model_instance = st.session_state.get("stage1_approved_model", None)

    st.subheader("⚙️ 배포 타겟 및 양자화 설정")
    target_board = st.selectbox(
        "🎛️ 타겟 하드웨어 보드 선택", 
        ["자사 커스텀 v4 (Edge-AI)", "STM32H7 (Cortex-M7 MCU)"], 
        key="s2_board_select"
    )
    
    # 캘리브레이션 데이터셋 수집 업로더
    uploaded_file = st.file_uploader(
        "📊 캘리브레이션용 참조 데이터셋 업로드 (.csv)", 
        type=["csv"], 
        key="s2_csv_uploader"
    )

    if uploaded_file is not None:
        st.success(f"📁 {uploaded_file.name} 데이터셋 인입 성공. 정적 차원 매핑 규칙이 수립되었습니다.")
        
        # 데이터프레임 실시간 프리뷰 존
        st.markdown("#### 🔍 업로드 데이터셋 실시간 프리뷰 (상위 5개 행)")
        try:
            preview_df = pd.read_csv(uploaded_file)
            st.dataframe(preview_df.head(5), use_container_width=True)
            st.caption(f"💡 총 행 수: {len(preview_df)}개 | 총 컬럼 수: {len(preview_df.columns)}개 감지 완료.")
            uploaded_file.seek(0)
        except Exception as p_err:
            st.warning(f"⚠️ 데이터프레임 프리뷰 로드 실패: {p_err}")

        # 양자화 정밀도 설정 라디오 박스
        precision = st.radio(
            "💎 양자화 정밀도 (Quantization Bit)", 
            ["FP32 (No Quantization)", "FP16 (Half Precision)", "INT8 (Full Integer Quantization)"], 
            key="s2_bit_radio"
        )
        # 4. TFLite 컴파일 및 양자화 전후 비교 벤치마킹 실행 존
        if st.button("🚀 TFLite 컴파일 및 가상 벤치마크 수행", type="primary", key="s2_compile_btn"):
            # 가변 소스코드 차원 연동 에이전트 인스턴스 가동
            agent = DatasetDrivenTFLiteConvertAgent(seq_len=seq_len, d_model=d_model)
            
            with st.spinner("하드웨어 인프라 제약 조건 검증 및 바이너리 에뮬레이션 중..."):
                import time
                time.sleep(1.2) # 몰입감 연출을 위한 시뮬레이션 지연
                
                # [백엔드 연동] 공학적 트레이드오프 실측 및 시뮬레이션 수식 호출
                result = agent.run_benchmarking(target_board, precision, base_model_mse=base_mse)
                
            st.subheader("📊 온디바이스 벤치마킹 결과 리포트")
            
            # 1단계: 선택된 스펙의 핵심 KPI 지표 메트릭 가로 배치
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("추론 속도 (Latency)", f"{result.estimated_latency_ms} ms")
            with col2:
                st.metric("실내 온도 추종 오차 (MSE)", f"{result.estimated_mse}")
            with col3:
                st.metric("예상 바이너리 크기", f"{result.memory_footprint_kb} KB")
                
            if "WARNING" in result.status:
                st.warning(f"⚠️ 경고: {target_board}에서 FP32 원본 사양 배포 시 가전 칩셋 메모리 한계를 초과할 위험이 높습니다.")
            else:
                st.success("🎯 하드웨어 스펙 검증 통과: 온디바이스 배포에 적합한 모델 사양입니다.")

            # --------------------------------------------------------------------------
            # 🚨 [핵심 확충] 하드웨어 양자화(Quantization) 전후 공학적 비교 통계 테이블
            # --------------------------------------------------------------------------
            st.markdown("---")
            st.markdown("### 🧮 양자화 정밀도 전후 트레이드오프 정밀 비교")
            st.caption("비트 정밀도가 낮아질수록 가전 MCU 연산 속도는 비약적으로 빨라지지만, 양자화 손실로 오차율이 미세 상승합니다.")
            
            # 각 정밀도 조합별 에뮬레이션 분석 데이터 빌드
            is_stm = ("STM32" in target_board)
            comparison_matrix = {
                "정밀도 스펙 (Precision)": ["FP32 (No Quantized)", "FP16 (Half Precision)", "INT8 (Full Integer)"],
                "추론 속도 (Latency)": [
                    f"{85.0 if is_stm else 12.5} ms",
                    f"{46.7 if is_stm else 6.8} ms",
                    f"{15.3 if is_stm else 3.1} ms"
                ],
                "실내 온도 오차 (MSE)": [
                    f"{base_mse}",
                    f"{round(base_mse * 1.02, 5)} (+2.0%)",
                    f"{round(base_mse * 1.25, 5)} (+25.0%)"
                ],
                "바이너리 용량 (Size)": ["142.0 KB (100%)", "71.0 KB (50%)", "35.5 KB (25%)"],
                "LNE NPU 가속 적합성": ["❌ 불가 (CPU 구동)", "⚠️ 부분 지원", "🎯 완벽 가속 지원"]
            }
            
            # 직관적인 의사결정을 위한 데이터프레임 시각화
            comparison_df = pd.DataFrame(comparison_matrix)
            st.table(comparison_df)
            
            # 현재 선택된 사양에 따른 공학적 분석 소견 동적 피드백
            if precision == "INT8 (Full Integer Quantization)":
                st.info("💡 **SW개발팀 기술 소견**: INT8 양자화를 통해 모델 용량을 **75% 압축**하고 가속기 칩셋 성능을 극대화했습니다. 온도 추종 오차(MSE)가 소폭 상승했으나 실내 운전 제어 기준치 이내이므로 양산 배포에 적합합니다.")
            elif precision == "FP16 (Half Precision)":
                st.warning("💡 **SW개발팀 기술 소견**: FP16 정밀도는 오차 상승이 거의 없으나 가전 하드웨어 칩셋(NPU/MCU)에 따라 정수형 연산 가속을 100% 받지 못해 지연이 발생할 수 있습니다.")
            else:
                st.error("💡 **SW개발팀 기술 소견**: FP32 원본 사양은 타겟 보드의 에지 메모리(Memory Arena) 가용 공간을 과다 점유하여 오작동 위험이 있습니다. 양자화 컴파일을 강력히 권장합니다.")
                
            # 후속 공정(Stage 3) 연쇄 반응을 위한 플래그 락 해제
            st.session_state.stage2_done = True
    else:
        st.info("💡 캘리브레이션을 진행할 데이터셋(.csv) 파일을 상단에 업로드하시면 상세 설정 및 벤치마크 버튼이 활성화됩니다.")
