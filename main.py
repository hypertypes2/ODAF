import os
import sys
import importlib

# 텐서플로우와 구글 SDK 간의 Protobuf 런타임 버전 버그를 원천 차단하는 환경변수 락
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
from session import ODAFSessionState

# --------------------------------------------------------------------------
# 🔄 [캐시 무효화 패치] 구버전 임포트 꼬임 및 소스코드 변경 실시간 반영 규칙
# --------------------------------------------------------------------------
for module_name in list(sys.modules.keys()):
    if "pipeline_stages" in module_name:
        importlib.reload(sys.modules[module_name])

# 강제 리로드 세션 확보 후 정상 임포트 가동
from pipeline_stages.stage0_data_analysis.ui import render_stage0
from pipeline_stages.stage1_model_design.ui import render_stage1
from pipeline_stages.stage2_tflite_convert.ui import render_stage2_interface

# --------------------------------------------------------------------------
# 🎛️ 대시보드 마스터 환경 정의
# --------------------------------------------------------------------------
st.set_page_config(page_title="ODAF 통합 파이프라인", layout="centered")

# 전사 세션 상태 및 공정 플래그 초기화
ODAFSessionState.initialize()

# --------------------------------------------------------------------------
# 🛰️ [UI 대개혁] 좌측 사이드바 수직 타임라인 링크 노드 시스템
# --------------------------------------------------------------------------
with st.sidebar:
    st.image("https://icons8.com", width=50)
    st.title("🛰️ ODAF 관제탑")
    st.caption("공정 링크를 확인하고 제어 구역으로 진입하십시오.")
    st.markdown("---")
    
    st.subheader("🏁 파이프라인 링크 트랙")
    
    # 각 공정의 내부 진행 규칙 데이터 추출
    s0_done = st.session_state.get("stage0_done", False)
    s1_done = st.session_state.get("stage1_approved", False)
    s2_done = st.session_state.get("stage2_done", False)
    
    # 1. 동적 노드(동그라미 마커) 그래픽 상태 연산
    # 완료 시 완전히 색칠된 노드(🟢), 미완료 시 빈 노드(⚪), 현재 선택 상태(🔹)
    node_s0 = "🟢" if s0_done else "⚪"
    node_s1 = "🟢" if s1_done else "⚪"
    node_s2 = "🟢" if s2_done else "⚪"
    
    # 2. 수직 링크 선 표기를 위한 마크다운 렌더링 존
    # 각 단계의 동그라미 노드 사이를 수직선(⋮) 기호로 이어 흐름을 가시화합니다.
    st.markdown(f"### {node_s0} Step 0: 데이터 분석")
    st.markdown("<div style='padding-left: 11px; color: #888; margin-top: -10px; margin-bottom: -10px;'>⋮</div>", unsafe_allow_html=True)
    
    st.markdown(f"### {node_s1} Step 1: 제약 조건 검증")
    st.markdown("<div style='padding-left: 11px; color: #888; margin-top: -10px; margin-bottom: -10px;'>⋮</div>", unsafe_allow_html=True)
    
    st.markdown(f"### {node_s2} Step 2: TFLite 벤치마크")
    
    st.markdown("---")
    st.subheader("⚙️ 관제 구역 강제 진입 선택")
    
    # 실제 화면 분기를 누르기 위한 내비게이션 드롭다운 메뉴 매핑
    menu_options = [
        "📊 Step 0: 데이터 관제 구역",
        "🧠 Step 1: 선행연구 최적화 구역",
        "⚙️ Step 2: SW개발 컴파일 구역"
    ]
    
    selected_menu = st.selectbox(
        "이동할 배포 공정 선택",
        options=menu_options,
        index=0,
        help="상단의 타임라인 노드 색칠 현황을 확인한 후 적절한 작업 구역을 지정하십시오."
    )
    
    st.markdown("---")
    st.caption("On-Device 배포 규격화 가이드 v4.8")

# --------------------------------------------------------------------------
# 🔄 [화면 분기 제어] 사이드바 맵 및 셀렉터 클릭에 따른 공정 화면 출력
# --------------------------------------------------------------------------
if selected_menu == "📊 Step 0: 데이터 관제 구역":
    st.title("❄️ ODAF 시스템 - 데이터 관제 존")
    render_stage0()

elif selected_menu == "🧠 Step 1: 선행연구 최적화 구역":
    st.title("❄️ ODAF 시스템 - 선행연구 존")
    if s0_done:
        render_stage1()
    else:
        st.warning("🔒 접근 제한: Step 0 [원천 데이터 무결성 검증] 버튼을 실행하여 노드를 먼저 색칠하십시오.")

elif selected_menu == "⚙️ Step 2: SW개발 컴파일 구역":
    st.title("❄️ ODAF 시스템 - SW개발 배포 존")
    if s1_done:
        render_stage2_interface()
    else:
        st.warning("🔒 접근 제한: Step 1 선행연구팀 심사 완료 후, 하단 [최종 승인 체크박스]를 체크하여 이관 링크 노드를 개방하십시오.")
