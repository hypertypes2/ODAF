import streamlit as st
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any

# =========================================================================
# [부서 1: 대학 / 선행연구팀] 데이터 규격 스키마
# =========================================================================

class Stage0DataAnalysisSchema(BaseModel):
    """0번 탭: 데이터 분석 단계 스키마 (현재는 단순 패스용 뼈대 유지)"""
    is_passed: bool = Field(True, description="데이터 분석 단계 통과 여부")

class Stage1ModelDesignSchema(BaseModel):
    """
    1번 탭: 모델 설계 및 LNE 제약 검증 스키마
    """
    model_name: str = Field(..., min_length=1, description="자율운전 AI 제어 모델 명칭")
    base_model_type: str = Field(..., description="원본 AI 알고리즘 형태")
    raw_weight_mb: float = Field(..., ge=0.1, le=500.0, description="원본 가중치 파일 크기")
    used_operators: List[str] = Field(..., min_items=1, description="모델 구성 TFLite 연산자 목록")

    @field_validator('used_operators')
    @classmethod
    def check_fatal_unsupported_ops(cls, ops: List[str]) -> List[str]:
        """LNE 가속기가 아예 지원하지 않는 완전 불가능 오퍼레이터 차단 검증"""
        fatal_ops = ["GeLU", "Custom_Exp", "MHA", "LayerNormalization"]
        for op in ops:
            if op in fatal_ops:
                raise ValueError(f"하드웨어 치명적 오류: [{op}] 연산자는 자사 LNE 가속기 컴파일러에서 구현 불가능합니다.")
        return ops

# =========================================================================
# [전사 공통] On-Device Air-Flow (ODAF) 세션 상태 및 흐름 관리자
# =========================================================================

class ODAFSessionState:
    """Streamlit의 영속성 세션을 부서별 락(Lock) 구조에 맞춰 동기화하는 클래스"""
    
    @staticmethod
    def initialize():
        """시스템 최초 가동 시 5대 공정 탭의 실행 제어 플래그와 데이터 저장소 초기화"""
        if "stage0_done" not in st.session_state: st.session_state.stage0_done = False
        if "stage1_done" not in st.session_state: st.session_state.stage1_done = False
        if "stage2_done" not in st.session_state: st.session_state.stage2_done = False
        if "stage3_done" not in st.session_state: st.session_state.stage3_done = False

        if "stage0_data" not in st.session_state: st.session_state.stage0_data = None
        if "stage1_data" not in st.session_state: st.session_state.stage1_data = None
        if "stage2_data" not in st.session_state: st.session_state.stage2_data = None
        if "stage3_data" not in st.session_state: st.session_state.stage3_data = None

    @staticmethod
    def sync_stage(stage_idx: int, payload: Dict[str, Any]) -> tuple[bool, str]:
        """화면에서 입력 버튼을 눌렀을 때 Pydantic 스키마로 데이터를 정밀 검증 후 동기화"""
        try:
            if stage_idx == 0:
                validated = Stage0DataAnalysisSchema(**payload)
                st.session_state.data_analysis_data = validated.model_dump() # 💡 명칭 일치 보정
                st.session_state.stage0_done = True
                return True, "✅ [선행연구팀] 데이터 분석 단계 연동 패스 완료."

            elif stage_idx == 1:
                if not st.session_state.stage0_done:
                    return False, "🔒 이전 단계를 먼저 완료해 주세요."
                    
                validated = Stage1ModelDesignSchema(**payload)
                st.session_state.stage1_data = validated.model_dump()
                st.session_state.stage1_done = True
                
                st.session_state.stage2_done = False
                st.session_state.stage3_done = False
                st.session_state.stage2_data = None
                st.session_state.stage3_data = None
                return True, "✅ [선행연구팀] AI 모델 설계 정보 검증 및 하드웨어 제어 동기화 완료!"

            return False, "존재하지 않는 파이프라인 단계 번호입니다."
        except Exception as e:
            return False, f"❌ 검증 실패: {str(e)}"
