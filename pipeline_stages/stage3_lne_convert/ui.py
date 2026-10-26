from __future__ import annotations

import streamlit as st

from session import ODAFSessionState
from .agent import Stage3LNEConvertAgent


def _compact_parsed_summary(parsed: dict) -> dict:
    return {
        'qpoint': parsed.get('qpoint'),
        'actual_delta': parsed.get('actual_delta'),
        'max_final_output_abs_error': parsed.get('max_final_output_abs_error'),
        'validation_summary_count': parsed.get('validation_summary_count'),
        'cpu_fallback_detected': parsed.get('cpu_fallback_detected'),
        'unsupported_operators': parsed.get('unsupported_operators', [])[:5],
        'lne_contract_checks': parsed.get('lne_contract_checks', {}),
    }


def render_stage3_interface() -> None:
    st.header("3️⃣ Stage 3: LNE Convert + SPIKE Validation")
    st.caption("Windows Local ODAF가 산출물을 준비하고, 원격 Linux VM Worker가 LNE/SPIKE 작업을 수행합니다.")

    stage2_summary = st.session_state.get('stage2_conversion_summary', {})
    if not stage2_summary:
        st.warning("Stage 2 산출물이 필요합니다.")
        return

    current_config = st.session_state.get('remote_worker_config', {})
    default_model_options = {
        'INT8 TFLite': st.session_state.get('stage2_tflite_path', ''),
        'FP32 TFLite': st.session_state.get('stage2_fp32_tflite_path', ''),
    }

    tab_connection, tab_lne, tab_spike = st.tabs(["🔌 Connection", "⚙️ LNE Convert", "🧪 SPIKE Simulation"])

    with tab_connection:
        with st.expander("🛰️ 원격 Linux VM Worker 설정", expanded=True):
            ssh_host = st.text_input("SSH Host", value=current_config.get('ssh_host', 'heetae-chung-lne-odaf.bee0.lge.com'), key='stage3_ssh_host')
            ssh_user = st.text_input("SSH User", value=current_config.get('ssh_user', current_config.get('ssh_username', 'worker')), key='stage3_ssh_user')
            ssh_port = st.number_input("SSH Port", min_value=1, max_value=65535, value=int(current_config.get('ssh_port', 59349)), key='stage3_ssh_port')
            ssh_key_path = st.text_input("SSH Key Path (optional)", value=current_config.get('ssh_key_path', ''), key='stage3_ssh_key_path')
            remote_ddk_root = st.text_input("Remote DDK Root", value=current_config.get('remote_ddk_root', '/home/worker/lne_ddk-lneddk_3.0-2609'), key='stage3_remote_ddk_root')
            lne_target = st.text_input("Target", value=current_config.get('lne_target', 'dqc'), key='stage3_lne_target')
            validation_set = st.number_input("Validation Set", min_value=1, max_value=10000, value=int(current_config.get('validation_set', 10)), key='stage3_validation_set')
            quant_loop = st.number_input("Quant Loop", min_value=1, max_value=10000, value=int(current_config.get('quant_loop', 10)), key='stage3_quant_loop')
            compile_timeout = st.number_input("Compile Timeout (sec)", min_value=10, max_value=7200, value=int(current_config.get('compile_timeout', 600)), key='stage3_compile_timeout')
            spike_timeout = st.number_input("SPIKE Timeout (sec)", min_value=10, max_value=7200, value=int(current_config.get('spike_timeout', 600)), key='stage3_spike_timeout')
            execution_mode = st.selectbox("Execution Mode", options=['remote', 'dry-run'], index=0 if current_config.get('execution_mode', 'remote') == 'remote' else 1, key='stage3_execution_mode')
            max_abs_error_value = current_config.get('validation', {}).get('max_abs_error', '')
            max_abs_error_input = st.text_input("MAX ABS.Error Threshold (optional)", value=str(max_abs_error_value) if max_abs_error_value != '' else '', key='stage3_max_abs_error')
            allow_cpu_fallback = st.checkbox(
                "Allow CPU Fallback",
                value=bool(current_config.get('validation', {}).get('allow_cpu_fallback', False)),
                key='stage3_allow_cpu_fallback',
            )

            if st.button("설정 저장", key='stage3_save_config'):
                validation = {'allow_cpu_fallback': allow_cpu_fallback}
                if max_abs_error_input.strip():
                    validation['max_abs_error'] = float(max_abs_error_input)
                st.session_state.remote_worker_config = {
                    'ssh_host': ssh_host,
                    'ssh_user': ssh_user,
                    'ssh_username': ssh_user,
                    'ssh_port': int(ssh_port),
                    'ssh_key_path': ssh_key_path,
                    'remote_ddk_root': remote_ddk_root,
                    'remote_tflite_dir': f"{remote_ddk_root.rstrip('/')}/tflite",
                    'remote_raw_input_dir': f"{remote_ddk_root.rstrip('/')}/raw_input",
                    'remote_output_dir': f"{remote_ddk_root.rstrip('/')}/output",
                    'remote_lne_json_path': f"{remote_ddk_root.rstrip('/')}/lne.json",
                    'lne_target': lne_target,
                    'validation_set': int(validation_set),
                    'quant_loop': int(quant_loop),
                    'compile_timeout': int(compile_timeout),
                    'spike_timeout': int(spike_timeout),
                    'execution_mode': execution_mode,
                    'validation': validation,
                }
                st.success("Stage 3 설정을 저장했습니다.")

        if st.button("🔌 Connection Test", key='stage3_connection_test_button'):
            agent = Stage3LNEConvertAgent()
            result = agent.test_connection(st.session_state.get('remote_worker_config', {}))
            st.session_state.stage3_last_connection_test = result
            st.success("Connection Test 완료")

        connection_test = st.session_state.get('stage3_last_connection_test')
        if connection_test:
            st.subheader("🔌 Connection Test Result")
            st.json(connection_test)

    with tab_lne:
        st.markdown("### 🔧 LNE 변환 단계")
        st.caption("원격 VM에서 TFLite → LNE compile만 수행하고 결과를 별도 확인합니다.")
        agent = Stage3LNEConvertAgent()
        own_result = st.session_state.get('stage3_lne_result')
        if own_result:
            st.subheader("📌 최근 LNE 변환 결과")
            st.caption(f"Run ID: {own_result.get('run_id', '-')}")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Status", own_result.get('status', '-'))
            with col2:
                st.metric("Connection", own_result.get('connection', {}).get('status', '-'))
            with col3:
                st.metric("Compile", own_result.get('compile_analysis', {}).get('build_status', '-'))
            with col4:
                st.metric("SPIKE", own_result.get('spike_analysis', {}).get('spike_status', '-'))

            st.json({
                'status': own_result.get('status'),
                'validation': own_result.get('validation', {}),
                'compile_analysis': own_result.get('compile_analysis', {}),
                'parsed': _compact_parsed_summary(own_result.get('parsed', {})),
            })

            with st.expander("Compile log"):
                compile_obj = own_result.get('compile', {})
                compile_text = (compile_obj.get('stdout', '') or '') + '\n' + (compile_obj.get('stderr', '') or '')
                st.code(agent._summarize_log(compile_text, max_lines=20))

            if own_result.get('validation', {}).get('failures'):
                st.warning("LNE validation failures:")
                st.write(own_result['validation']['failures'])

        selected_model_label = st.radio(
            "모델 선택",
            options=list(default_model_options.keys()),
            horizontal=True,
            index=0,
        )

        local_inputs = {
            'saved_model_dir': st.session_state.get('stage2_saved_model_dir', ''),
            'fp32_tflite': st.session_state.get('stage2_fp32_tflite_path', ''),
            'int8_tflite': st.session_state.get('stage2_tflite_path', ''),
            'selected_tflite': default_model_options[selected_model_label],
            'calibration_folder': st.session_state.get('stage2_calibration_folder', ''),
            'raw_input_dir': st.session_state.get('stage2_raw_input_dir', ''),
            'reference_output': st.session_state.get('stage2_reference_output_path', ''),
        }
        st.subheader("📦 Stage 2 입력 산출물")
        st.json(local_inputs)

        if st.button("🧪 Dry Run", key='stage3_dry_run_btn'):
            agent = Stage3LNEConvertAgent()
            dry_run_config = dict(st.session_state.get('remote_worker_config', {}))
            dry_run_config['execution_mode'] = 'dry-run'
            output_dir = str(ODAFSessionState.stage_output_dir('stage3_lne_convert'))
            result = agent.run_stage3_pipeline(dry_run_config, local_inputs, st.session_state.get('stage1_refactored_code', ''), output_dir)
            st.session_state.stage3_result = result
            st.rerun()

        if st.button("🚀 LNE 변환 실행", type='primary', key='stage3_run_btn'):
            agent = Stage3LNEConvertAgent()
            output_dir = str(ODAFSessionState.stage_output_dir('stage3_lne_convert'))
            run_status = st.status("🚀 LNE 변환을 시작합니다...", state="running")

            def on_progress(phase: str, message: str) -> None:
                state = "complete" if phase in {"Finished"} else ("error" if phase in {"Blocked", "Connection failed"} else "running")
                run_status.update(label=f"{phase}: {message}", state=state)
                run_status.write(f"{phase}: {message}")

            result = agent.run_stage3_pipeline(
                st.session_state.get('remote_worker_config', {}),
                local_inputs,
                st.session_state.get('stage1_refactored_code', ''),
                output_dir,
                progress_callback=on_progress,
                phase='lne',
            )
            st.session_state.stage3_lne_result = result
            st.rerun()

        if not own_result:
            st.info("아직 실행 결과가 없습니다. Dry Run 또는 LNE 변환 실행을 눌러 결과를 생성하세요.")

    with tab_spike:
        st.markdown("### 🧪 SPIKE 시뮬레이션 단계")
        st.caption("이미 LNE compile가 끝난 상태에서 SPIKE 검증만 별도 수행합니다.")
        agent = Stage3LNEConvertAgent()
        spike_result = st.session_state.get('stage3_spike_result')
        if spike_result:
            st.subheader("📌 최근 SPIKE 결과")
            st.caption(f"Run ID: {spike_result.get('run_id', '-')}")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Status", spike_result.get('status', '-'))
            with col2:
                st.metric("Connection", spike_result.get('connection', {}).get('status', '-'))
            with col3:
                st.metric("Compile", spike_result.get('compile_analysis', {}).get('build_status', '-'))
            with col4:
                st.metric("SPIKE", spike_result.get('spike_analysis', {}).get('spike_status', '-'))

            st.json({
                'status': spike_result.get('status'),
                'validation': spike_result.get('validation', {}),
                'compile_analysis': spike_result.get('compile_analysis', {}),
                'spike_analysis': spike_result.get('spike_analysis', {}),
                'parsed': _compact_parsed_summary(spike_result.get('parsed', {})),
            })

            with st.expander("SPIKE log"):
                spike_obj = spike_result.get('spike', {})
                spike_text = (spike_obj.get('stdout', '') or '') + '\n' + (spike_obj.get('stderr', '') or '')
                st.code(agent._summarize_log(spike_text, max_lines=20))

            if spike_result.get('validation', {}).get('failures'):
                st.warning("SPIKE validation failures:")
                st.write(spike_result['validation']['failures'])

        if st.button("🚀 SPIKE 실행", key='stage3_spike_only_btn'):
            agent = Stage3LNEConvertAgent()
            output_dir = str(ODAFSessionState.stage_output_dir('stage3_lne_convert'))
            result = agent.run_stage3_pipeline(
                st.session_state.get('remote_worker_config', {}),
                local_inputs,
                st.session_state.get('stage1_refactored_code', ''),
                output_dir,
                phase='spike',
            )
            st.session_state.stage3_spike_result = result
            st.rerun()

        if not spike_result:
            st.info("아직 SPIKE 결과가 없습니다. LNE 변환이 성공한 뒤 SPIKE를 실행하세요.")

        parsed = spike_result.get('parsed', {}) or {} if spike_result else {}
        if spike_result and spike_result.get('status') == 'NOT_CONFIGURED':
            st.warning("Stage 3 실행 전 확인이 필요한 항목이 있습니다.")
            st.json(spike_result)
            return

        if spike_result:
            st.subheader("🔎 실행 결과")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Overall", spike_result.get('validation', {}).get('overall_status', spike_result.get('status', '-')))
            with col2:
                st.metric("Build", spike_result.get('validation', {}).get('build_status', '-'))
            with col3:
                st.metric("SPIKE", spike_result.get('validation', {}).get('spike_status', '-'))
            with col4:
                st.metric("Verdict", spike_result.get('validation', {}).get('verdict', '-'))

            if parsed.get('qpoint') is not None:
                st.caption(f"Q-point: {parsed.get('qpoint')} | Actual delta: {parsed.get('actual_delta')} | Summary count: {parsed.get('validation_summary_count', 0)}")

            if parsed.get('lne_contract_checks'):
                st.markdown("### LNE Guide Contract Check")
                st.json(parsed['lne_contract_checks'])

            if spike_result.get('validation', {}).get('verdict') not in {'FAIL', 'DRY_RUN'}:
                if st.button("✅ Stage 3 완료 - Stage 4로 진행", key='stage3_continue_btn'):
                    st.session_state.stage3_verified = True
                    st.session_state.stage3_done = True
                    st.session_state.selected_stage = 4
                    st.success("Stage 3 완료! 사이드바에서 Stage 4로 이동하세요.")
            else:
                st.warning("Stage 3 결과가 PASS가 아닙니다. Stage 1/2 수정 또는 VM 설정 점검 후 재실행하세요.")
