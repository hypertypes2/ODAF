import streamlit as st
from pipeline_stages.stage1_model_design.agent import ModelDesignAgent

def render_stage1():
    st.header("1️⃣ Stage 1: 모델 설계 및 LNE 제약 검증")
    st.caption("선행연구팀 주관 - 온디바이스 하드웨어 가속용 정적 루프 강제 주입 단계")

    default_code = "model.add(layers.GRU(64, return_sequences=True))\nmodel.add(layers.Dense(5))"
    source_code_input = st.text_area("💻 훈련 완료된 파이썬 소스코드 입력", value=default_code, height=120, key="s1_raw_code_area")

    # 세션 상태 사전 바인딩 방어선
    if "s1_pipeline_data" not in st.session_state:
        st.session_state.s1_pipeline_data = None

    if st.button("🔧 온디바이스 최적화 코드 리팩터링 수행", type="secondary", key="s1_trigger_btn"):
        with st.spinner("AI 에이전트가 코드를 분석하고 하드웨어 제약 조건을 매핑 중..."):
            agent = ModelDesignAgent()
            # 원스톱 파이프라인 연산 호출 및 세션 캐싱
            st.session_state.s1_pipeline_data = agent.run_stage1_pipeline(source_code_input)
            
            # 리팩터링 완료된 결과 코드를 메인 스위칭 플래그에 바인딩
            res = st.session_state.s1_pipeline_data
            if res["refactored"]:
                st.session_state.stage1_refactored_code = res["refactored"]["refactored_code"]
            else:
                st.session_state.stage1_refactored_code = source_code_input

    # --------------------------------------------------------------------------
    # 📊 분석 결과 뷰어 레이아웃 전개 (Single Column Track)
    # --------------------------------------------------------------------------
    if st.session_state.s1_pipeline_data is not None:
        res_data = st.session_state.s1_pipeline_data
        
        # 1. 원본 소스코드 연산자 분석 리포트 존
        st.subheader("🔍 원본 소스코드 하드웨어 제약 및 오퍼레이터 분석")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("유추된 가전 모델명", res_data["analysis"].get("model_name", "Unknown"))
            st.metric("지배적 알고리즘 구조", res_data["analysis"].get("base_model_type", "Unknown"))
        with col2:
            st.metric("예상 가중치 파일 용량", f"{res_data['analysis'].get('raw_weight_mb', 0.0)} MB")
            unroll_status = "⚠️ 누락 (컴파일 불가)" if not res_data["analysis"].get("has_unroll_true", False) else "🟢 확보"
            st.metric("하드웨어 정적 루프", unroll_status)
        st.markdown("**검출된 모델 구성 연산자(Operators) 풀:**")
        st.code(f"{res_data['analysis'].get('parsed_operators', [])}")
        
        # 2. 컴파일러 제약 결과 분기 피드백
        if res_data["simulation"]["status"] == "FAILED":
            st.error(f"❌ 가속 컴파일 제약 위반 경고:\n{res_data['simulation']['error_message']}")
        else:
            st.success("🟢 LNE 가속기 가상 컴파일 무결성 검증 통과! (unroll 최적화 조건 충족 완료)")

        # 3. 보정 제안 소스코드 렌더링
        st.markdown("---")
        st.subheader("📝 온디바이스 배포용 자동 보정 소스코드")
        st.code(st.session_state.stage1_refactored_code, language="python")
        st.info("💡 타깃 보드 가속 연산 규격(TFLM) 정렬을 위해 unroll=True 옵션이 영속 주입되었습니다.")
        
        # 4. 수학적 동등성 및 무결성 검증 리포트 존
        st.markdown("---")
        st.subheader("📐 알고리즘 수학적 증명")
        st.markdown(res_data["mathematical_equivalence"])
        
        # 5. SW개발팀(Stage 2) 이관을 위한 핵심 체크박스 제어
        st.markdown("---")
        st.markdown("### 🔒 공정 승인 및 SW개발팀 이관 제어")
        st.session_state.stage1_approved = st.checkbox(
            "✅ 위 온디바이스 최적화 보정 사양 및 수학적 동등성을 최종 승인하고 2️⃣ Stage 2 공정으로 이관합니다.",
            value=st.session_state.get("stage1_approved", False),
            key="s1_approve_check"
        )
