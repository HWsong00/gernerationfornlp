from langgraph.graph import StateGraph, END

from nodes.state import MCQState

from nodes.classifier import router_node, route_decision

from nodes.retriever import retrieve_node

from nodes.solver import ko_history_solver_node, general_solver_node, recovery_node

from nodes.parser import parser_node

# nodes/state.py 에서 reranker, ensemble_retriever 필드는 삭제된 상태여야 합니다.

def create_mcq_workflow(ensemble_retriever, reranker):
    workflow = StateGraph(MCQState)

    workflow.add_node("classify", router_node)
    
    # 람다 방식 유지 (깔끔합니다!)
    workflow.add_node("retrieve", lambda s: retrieve_node(s, ensemble_retriever, reranker))
    
    workflow.add_node("ko_history_solve", ko_history_solver_node)
    workflow.add_node("general_solve", general_solver_node)
    workflow.add_node("parse", parser_node)
    workflow.add_node("recovery", recovery_node)

    workflow.set_entry_point("classify")

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

    # [디테일 수정] x["final_answer"]가 "N/A"이거나 None인 경우를 모두 체크
    workflow.add_conditional_edges(
        "parse",
        lambda x: "end" if x.get("final_answer") and x["final_answer"] != "N/A" else "recovery",
        {
            "end": END,
            "recovery": "recovery"
        }
    )

    workflow.add_edge("recovery", END)

    return workflow.compile()