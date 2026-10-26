from __future__ import annotations

import json

import streamlit as st

from .agent import ExecutiveSignAgent


def render_stage5_interface(session_manager=None) -> None:
    st.header("5️⃣ Stage 5: Executive Sign")
    st.caption("Stage 1~4의 결과를 종합해 최종 승인 리포트를 생성합니다.")

    if not st.session_state.get('stage4_done', False):
        st.warning('🔒 먼저 Stage 4를 완료해주세요.')
        return

    agent = ExecutiveSignAgent(session_manager=session_manager)
    report = agent.generate_approval_report(
        st.session_state.get('stage1_summary', {}),
        st.session_state.get('stage2_conversion_summary', {}),
        st.session_state.get('stage3_validation_summary', {}),
        st.session_state.get('stage4_simulation_summary', {}),
    )
    st.session_state.stage5_approval_summary = report.model_dump()
    st.session_state.stage5_data = report.model_dump()

    col1, col2 = st.columns(2)
    with col1:
        st.metric('Approval Score', f"{report.approval_score:.1f} / 100")
    with col2:
        st.write(report.approval_status)

    tabs = st.tabs(['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4', 'Recommendation'])
    with tabs[0]:
        st.json(report.stage1_summary)
    with tabs[1]:
        st.json(report.stage2_summary)
    with tabs[2]:
        st.json(report.stage3_summary)
    with tabs[3]:
        st.json(report.stage4_summary)
    with tabs[4]:
        st.info(report.final_recommendation)

    final_report_json = json.dumps(report.model_dump(), indent=2, ensure_ascii=False)
    st.download_button(
        label='📄 최종 승인 리포트 다운로드 (JSON)',
        data=final_report_json,
        file_name='ODAF_Executive_Sign_Report.json',
        mime='application/json',
        width='stretch',
    )

    if st.button('🎉 ODAF 파이프라인 완료', type='primary', width='stretch'):
        st.session_state.pipeline_complete = True
        st.session_state.stage5_done = True
        st.balloons()
        st.success('ODAF 전체 파이프라인이 완료되었습니다.')
