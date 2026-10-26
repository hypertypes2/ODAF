from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field


class ProductionApprovalReport(BaseModel):
    approval_score: float = Field(...)
    approval_status: str = Field(...)
    stage1_summary: Dict[str, Any] = Field(...)
    stage2_summary: Dict[str, Any] = Field(...)
    stage3_summary: Dict[str, Any] = Field(...)
    stage4_summary: Dict[str, Any] = Field(...)
    final_recommendation: str = Field(...)


class ExecutiveSignAgent:
    def __init__(self, session_manager=None) -> None:
        self.session_manager = session_manager

    def generate_approval_report(
        self,
        stage1_data: Dict[str, Any],
        stage2_data: Dict[str, Any],
        stage3_data: Dict[str, Any],
        stage4_data: Dict[str, Any],
    ) -> ProductionApprovalReport:
        approval_score = 100.0

        if stage2_data.get('accuracy_delta', 0) > 5:
            approval_score -= 10
        if stage3_data.get('verdict') == 'FAIL':
            approval_score -= 25
        elif stage3_data.get('verdict') == 'WARNING':
            approval_score -= 10
        if stage4_data.get('pass_rate', 0) < 100:
            approval_score -= 10

        if approval_score >= 90:
            approval_status = '✅ APPROVED - 즉시 양산 검토 가능'
        elif approval_score >= 75:
            approval_status = '⚠️ CONDITIONAL - 추가 확인 후 승인'
        else:
            approval_status = '❌ REJECTED - 선행 단계 보완 필요'

        final_recommendation = (
            f"Stage 1~4 결과를 종합한 승인 점수는 {approval_score:.1f}점입니다. "
            f"Stage 3 verdict={stage3_data.get('verdict', 'N/A')}, Stage 4 pass rate={stage4_data.get('pass_rate', 'N/A')}% 기준으로 "
            "원격 VM 검증 로그와 양산 시뮬레이션 결과를 함께 첨부해 최종 보고하는 것을 권장합니다."
        )

        return ProductionApprovalReport(
            approval_score=approval_score,
            approval_status=approval_status,
            stage1_summary=stage1_data,
            stage2_summary=stage2_data,
            stage3_summary=stage3_data,
            stage4_summary=stage4_data,
            final_recommendation=final_recommendation,
        )
