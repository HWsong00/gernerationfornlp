from nodes.state import MCQState
from utils.llm import get_llm

def router_node(state: MCQState):
    """
    ==== 과목 분류 및 RAG 여부 결정 노드 (OpenAI Format) =====
    """
    # 1. Baseline 테스트용 스킵 로직 (외부에서 설정된 경우 LLM 호출 안 함)
    # 변수명 변경: is_korean_history -> needs_knowledge
    if state.get("needs_knowledge") is not None:
        current_val = state["needs_knowledge"]
        print(f"⏩ [Router] 외부 설정값(지식 검색 필요: {current_val})이 감지되어 분류를 스킵합니다.")
        return {"needs_knowledge": current_val}

    llm = get_llm()
    
    print(f"🔍 [Router] 지식 검색 필요성 판단 시작 (ID: {state.get('id', 'unknown')})")

    # 2. OpenAI 형식의 메시지 구성
    # 프롬프트 내의 분류 라벨은 모델의 이해를 돕기 위해 유지하거나 더 범용적인 단어로 바꿀 수 있습니다.
    messages = [
        {
            "role": "system", 
            "content": (
                "당신은 과목 분류 전문가입니다. 주어진 문제가 구체적인 외부 지식(역사, 경제, 정치, 법률 등) "
                "검색이 필요한 문제인지 판단하세요.\n"
                "- 지식 검색이 꼭 필요한 경우: 'KNOWLEDGE_REQUIRED'\n"
                "- 일반 논리, 단순 독해, 상식으로 풀 수 있는 경우: 'GENERAL'\n"
                "결과는 반드시 'KNOWLEDGE_REQUIRED' 또는 'GENERAL' 중 한 단어로만 답하세요."
            )
        },
        {
            "role": "user", 
            "content": f"[지문]\n{state.get('paragraph', '')}\n\n[질문]\n{state.get('question', '')}"
        }
    ]
    
    # 3. 직접 호출
    response = llm.invoke(messages)
    result = response.content.strip().upper()
    
    # 결과 판단 로직 변경
    needs_knowledge = "KNOWLEDGE_REQUIRED" in result
    print(f"📊 [Router] 분류 결과: {'지식 검색 필요(RAG)' if needs_knowledge else '일반 독해'}")
    
    return {"needs_knowledge": needs_knowledge}

def route_decision(state: MCQState):
    """
    분기 결정 함수
    """
    # 변수명 변경 반영
    if state.get("needs_knowledge", False):
        return "retrieve"
    return "general_solve"