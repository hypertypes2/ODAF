import streamlit as st
import pandas as pd
import numpy as np

def render_stage0():
    st.header("0️⃣ Stage 0: 실시간 온디바이스 시계열 데이터 분석")
    st.caption("선행연구팀 주관 - 원천 센서 데이터 정규화 및 피처 타깃 도메인 분석")

    # 가상 시계열 실시간 대시보드 플롯
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['실내온도 센서', '토출구 압력', '기류 순환속도']
    )
    st.line_chart(chart_data)

    # 부서간 락 제어를 위한 상태 업데이트
    if st.button("📊 원천 데이터 무결성 검증 및 전처리 완료", key="s0_commit_btn"):
        st.session_state.stage0_done = True
        st.success("🎯 원천 데이터셋 분산 및 누락치 처리 통과. 1단계 공정이 해제되었습니다.")
