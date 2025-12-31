from langgraph.graph import StateGraph, END
from nodes.state import MCQState
from nodes.classifier import router_node, route_decision
from nodes.retriever import retrieve_node
from nodes.solver import ko_history_solver_node, general_solver_node, recovery_node
from nodes.parser import parser_node

def create_mcq_workflow(ensemble_retriever):
    """
    MCQ 문제 풀이를 위한 전체 워크플로우 조립
    """
    workflow = StateGraph(MCQState)

    # 1. 노드 등록
    workflow.add_node("classify", router_node)
    
    # retriever는 외부 객체(ensemble_retriever)를 주입받아야 하므로 lambda 활용
    workflow.add_node("retrieve", lambda s: retrieve_node(s, ensemble_retriever))
    
    workflow.add_node("ko_history_solve", ko_history_solver_node)
    workflow.add_node("general_solve", general_solver_node)
    workflow.add_node("parse", parser_node)
    workflow.add_node("recovery", recovery_node)

    # 2. 에지(흐름) 연결
    workflow.set_entry_point("classify")

    # [조건부 에지] 한국사 여부에 따라 RAG 여부 결정
    workflow.add_conditional_edges(
        "classify",
        route_decision,
        {
            "retrieve": "retrieve",
            "general_solve": "general_solve"
        }
    )

    workflow.add_edge("retrieve", "ko_history_solve")
    workflow.add_edge("ko_history_solve", "parse")
    workflow.add_edge("general_solve", "parse")

    # [조건부 에지] 정답 추출 실패 시 Recovery 노드로 이동
    workflow.add_conditional_edges(
        "parse",
        lambda x: "end" if x["final_answer"] != "N/A" else "recovery",
        {
            "end": END,
            "recovery": "recovery"
        }
    )

    workflow.add_edge("recovery", END)

    # 3. 컴파일
    return workflow.compile()