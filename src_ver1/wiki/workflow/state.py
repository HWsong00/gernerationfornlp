from typing import TypedDict, List


class ProblemState(TypedDict):
    """문제 풀이 워크플로우의 상태"""
    # 입력
    context: str  # 지문
    question: str  # 문제
    choices: List[str]
    row_id: str  # 문제 ID

    # 중간 결과
    problem_type: str  # 문제 유형 
    needs_external_knowledge: bool  # 외부 지식 필요 여부
    rag_context: str  # RAG로 검색된 보강 자료
    reasoning: str  # 판단 근거

    # 최종 결과
    final_answer: str  # 최종 답변
    confidence: float  # 신뢰도