from __future__ import annotations

import io
import json

import matplotlib.pyplot as plt
import streamlit as st

from session import ODAFSessionState
from .agent import ProductionSimulationAgent


def render_stage4_interface() -> None:
    st.header("4️⃣ Stage 4: Production Simulation")
    st.caption("Test Case, Case Study, Simulation, Visualization을 통해 양산 시뮬레이션 결과를 검토합니다.")

    stage3_summary = st.session_state.get('stage3_validation_summary', {})
    stage3_verified = st.session_state.get('stage3_verified', False)
    tflite_path = st.session_state.get('stage2_tflite_path') or st.session_state.get('stage2_fp32_tflite_path') or ''
    source_code = st.session_state.get('stage1_refactored_code') or st.session_state.get('stage1_original_code') or ''

    if not tflite_path:
        st.warning('Stage 2 TFLite 모델이 필요합니다.')
        return
    if not stage3_verified and not stage3_summary:
        st.warning('Stage 3 결과가 필요합니다.')
        return

    if st.button('🚀 양산 시뮬레이션 실행', type='primary', key='stage4_run_btn'):
        agent = ProductionSimulationAgent(
            seq_len=int(st.session_state.get('parsed_seq_len', 12)),
            d_model=int(st.session_state.get('parsed_d_model', 7)),
        )
        result = agent.run_full_simulation_pipeline(source_code, tflite_path)
        st.session_state.stage4_result = result.model_dump()
        st.session_state.stage4_simulation_summary = {
            'overall_status': result.overall_status,
            'pass_rate': result.case_study['pass_rate'],
            'scenario_count': result.simulation_summary['scenario_count'],
            'task_type': result.task_type,
            'confidence': result.confidence,
        }
        st.session_state.stage4_data = st.session_state.stage4_simulation_summary
        st.session_state.stage4_done = True
        st.rerun()

    result = st.session_state.get('stage4_result')
    if not result:
        st.info('Stage 4 실행 후 결과가 표시됩니다.')
        return

    tabs = st.tabs(['🧪 Test Case', '📚 Case Study', '🖥️ Simulation', '📈 Visualization'])

    with tabs[0]:
        for test_result in result['test_results']:
            with st.expander(test_result['scenario_name'], expanded=True):
                st.caption(test_result['description'])
                st.json(test_result['test_cases'])

    with tabs[1]:
        st.json(result['case_study'])
        for rec in result['recommendations']:
            st.info(rec)

    with tabs[2]:
        st.json(result['simulation_summary'])
        st.metric('Overall Status', result['overall_status'])
        report_text = json.dumps(result, ensure_ascii=False, indent=2)
        st.download_button('📄 Simulation Report Download', report_text, 'stage4_simulation_report.json', 'application/json')

    with tabs[3]:
        buffer = io.BytesIO()
        scenario_names = [entry['scenario_name'] for entry in result['test_results']]
        pass_counts = [sum(1 for case in entry['test_cases'] if case['status'] == 'PASS') for entry in result['test_results']]
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.bar(scenario_names, pass_counts, color='royalblue')
        axis.set_ylabel('PASS Count')
        axis.set_title('Scenario PASS Summary')
        plt.tight_layout()
        plt.savefig(buffer, format='png', dpi=120)
        buffer.seek(0)
        st.image(buffer.read(), width='stretch')
        plt.close(figure)

    if st.button('✅ Stage 4 완료 - Stage 5로 진행', key='stage4_continue_btn'):
        ODAFSessionState.mark_stage_complete(4, st.session_state.stage4_simulation_summary)
        st.success('Stage 4 완료! 사이드바에서 Stage 5로 이동하세요.')
