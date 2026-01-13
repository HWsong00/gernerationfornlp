"""
워크플로우 그래프 구성
"""

from typing import List
from langgraph.graph import StateGraph, END, START
from workflow.state import ProblemState
from workflow.nodes import (
    analyze_problem,
    retrieve_rag_context,
    solve_with_rag,
    solve_without_rag
)
from rag.workflow.state import route_by_rag_decision


def create_rag_decision_workflow():
    """RAG on/off 결정 워크플로우 생성"""

    workflow = StateGraph(ProblemState)

    # 노드 추가
    workflow.add_node("analyze_problem", analyze_problem)
    workflow.add_node("retrieve_rag", retrieve_rag_context)
    workflow.add_node("solve_with_rag", solve_with_rag)
    workflow.add_node("solve_without_rag", solve_without_rag)

    # 엣지 연결
    workflow.add_edge(START, "analyze_problem")

    # 조건부 라우팅: RAG 사용 여부 결정
    workflow.add_conditional_edges(
        "analyze_problem",
        route_by_rag_decision,
        {
            "use_rag": "retrieve_rag",
            "no_rag": "solve_without_rag"
        }
    )

    # RAG 경로: retrieve_rag -> solve_with_rag -> END
    workflow.add_edge("retrieve_rag", "solve_with_rag")
    workflow.add_edge("solve_with_rag", END)

    # No RAG 경로: solve_without_rag -> END
    workflow.add_edge("solve_without_rag", END)

    return workflow.compile()


def solve_problem(
    context: str,
    question: str,
    choices: List[str],
    row_id: str = "unknown"
) -> ProblemState:
    """
    문제를 풀고 결과 반환

    Args:
        context: 지문
        question: 문제
        choices: 선지 리스트
        row_id: 문제 ID

    Returns:
        ProblemState: 최종 상태 (답변 포함)
    """
    app = create_rag_decision_workflow()

    initial_state = {
        "context": context,
        "question": question,
        "choices": choices,
        "row_id": row_id,
        "problem_type": "",
        "needs_external_knowledge": False,
        "rag_context": "",
        "reasoning": "",
        "final_answer": "",
        "confidence": 0.0
    }

    result = app.invoke(initial_state)
    return result