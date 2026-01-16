from nodes.state import MCQState
from utils.llm import get_llm_client, MODEL_NAME

def router_node(state: MCQState):
    """
    ==== 과목 분류 및 RAG 여부 결정 노드 (Native OpenAI SDK) =====
    """
    # 1. 외부 설정값(Baseline 테스트 등) 우선 처리 로직
    if state.get("needs_knowledge") is not None:
        current_val = state["needs_knowledge"]
        print(f"⏩ [Router] 외부 설정값(needs_knowledge: {current_val})이 감지되어 분류를 스킵합니다.")
        return {"needs_knowledge": current_val}

    # 2. Native 클라이언트 로드
    client = get_llm_client()
    
    print(f"🔍 [Router] 지식 검색 필요성 판단 시작 (ID: {state.get('id', 'unknown')})")

    # 3. Native SDK 형식의 메시지 구성
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
    
    # 4. Native SDK 호출 (추상화 레이어 제거)
    # temperature=0을 설정하여 분류의 일관성을 확보합니다.
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0,  # 분류는 결정적이어야 함
        max_tokens=20   # 단어 하나만 필요하므로 토큰 낭비 방지
    )
    
    result = response.choices[0].message.content.strip().upper()
    
    # 5. 결과 판단 및 상태 업데이트
    needs_knowledge = "KNOWLEDGE_REQUIRED" in result
    print(f"📊 [Router] 분류 결과: {'지식 검색 필요(RAG)' if needs_knowledge else '일반 독해'}")
    
    return {"needs_knowledge": needs_knowledge}

def route_decision(state: MCQState):
    """
    분기 결정 함수 (LangGraph용)
    """
    if state.get("needs_knowledge", False):
        return "retrieve"
    return "general_solve"