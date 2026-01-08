"""
조건부 라우팅 함수
"""

from typing import Literal
from workflow.state import ProblemState


def route_by_rag_decision(state: ProblemState) -> Literal["use_rag", "no_rag"]:
    """
    문제 분석 결과에 따라 RAG 사용 여부 결정
    """
    needs_rag = state.get('needs_external_knowledge', True)

    if needs_rag:
        return "use_rag"
    else:
        return "no_rag"