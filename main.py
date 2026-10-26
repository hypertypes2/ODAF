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

# 강제 리로드 세션 확보 후 분기 지연 임포트로 안정 가동

# --------------------------------------------------------------------------
# 🎛️ 대시보드 마스터 환경 정의
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="ODAF 통합 파이프라인", 
    layout="wide",
    page_icon="🚀",
    initial_sidebar_state="expanded"
)

# 전사 세션 상태 및 공정 플래그 초기화
ODAFSessionState.initialize()
STAGES = ODAFSessionState.stage_definitions()


def can_navigate(stage_index: int) -> bool:
    if stage_index == 1:
        return True
    previous_stage_key = f"stage{stage_index - 1}_done"
    if stage_index == 4 and st.session_state.get("stage3_verified", False):
        return True
    return st.session_state.get(previous_stage_key, False)


def render_selected_stage(selected_stage: int) -> None:
    st.title("🚀 ODAF - On-Device AI Flow")

    if selected_stage == 1:
        st.markdown("### 🧠 Stage 1: Model Design")
        from pipeline_stages.stage1_model_design.ui import render_stage1
        render_stage1()
    elif selected_stage == 2:
        st.markdown("### ⚙️ Stage 2: TFLite Convert")
        if can_navigate(2):
            from pipeline_stages.stage2_tflite_convert.ui import render_stage2_interface
            render_stage2_interface()
        else:
            st.warning("🔒 먼저 Stage 1을 완료해주세요.")
    elif selected_stage == 3:
        st.markdown("### 🛰️ Stage 3: LNE Convert + SPIKE Validation")
        if can_navigate(3):
            from pipeline_stages.stage3_lne_convert.ui import render_stage3_interface
            render_stage3_interface()
        else:
            st.warning("🔒 먼저 Stage 2를 완료해주세요.")
    elif selected_stage == 4:
        st.markdown("### 🧪 Stage 4: Production Simulation")
        if can_navigate(4):
            from pipeline_stages.stage4_production_simulation.ui import render_stage4_interface
            render_stage4_interface()
        else:
            st.warning("🔒 먼저 Stage 3을 완료해주세요.")
    elif selected_stage == 5:
        st.markdown("### ✅ Stage 5: Executive Sign")
        if can_navigate(5):
            from pipeline_stages.stage5_executive_sign.ui import render_stage5_interface
            render_stage5_interface(session_manager=None)
        else:
            st.warning("🔒 먼저 Stage 4를 완료해주세요.")

# --------------------------------------------------------------------------
# 🛰️ [UI 대개혁] 좌측 사이드바 수직 타임라인 링크 노드 시스템
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("🚀 ODAF 관제탑")
    st.markdown("### **On-Device AI Flow**")
    st.caption("온디바이스 AI 배포 자동화 파이프라인")
    st.markdown("---")

    st.subheader("🏁 파이프라인 링크 트랙")
    for stage in STAGES:
        stage_done = st.session_state.get(f"{stage['key']}_done", False)
        if stage["index"] == 5:
            stage_done = stage_done or st.session_state.get("pipeline_complete", False)
        node = "🟢" if stage_done else "⚪"
        button_label = f"{node} Stage {stage['index']}: {stage['title']}"
        if st.button(
            button_label,
            key=f"nav_{stage['key']}",
            width="stretch",
            disabled=not can_navigate(stage["index"]),
        ):
            ODAFSessionState.set_selected_stage(stage["index"])

        if stage["index"] != STAGES[-1]["index"]:
            st.markdown("<div style='padding-left: 11px; color: #888; margin-top: -10px; margin-bottom: -10px;'>⋮</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("💡 **Tip**: 위 버튼을 클릭하여 각 단계로 이동하세요")
    st.caption("On-Device AI Flow v2.0")

# --------------------------------------------------------------------------
# 🔄 [화면 분기 제어] 사이드바 버튼 클릭에 따른 공정 화면 출력
# --------------------------------------------------------------------------
selected_stage = st.session_state.get("selected_stage", 1)
render_selected_stage(selected_stage)
