#########################################################################################################
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from nodes.state import MCQState
from utils.llm import get_llm

# 1. 템플릿 정의 (직접적인 ChatML 태그는 삭제)
ROUTER_SYS_TEMPLATE = SystemMessagePromptTemplate.from_template(
    """당신은 과목 분류 전문가입니다. 다음 문제가 "한국사" 과목의 문제인지 판단하세요.

[분류 기준]
- 한국사 지식이 꼭 필요한 문제: 'KOREAN_HISTORY'
- 세계사, 일반 논리, 문학, 단순 상식 등: 'GENERAL'

결과는 반드시 'KOREAN_HISTORY' 또는 'GENERAL' 중 한 단어로만 답하세요."""
)

ROUTER_HUMAN_TEMPLATE = HumanMessagePromptTemplate.from_template(
    """[지문]
{paragraph}

[질문]
{question}"""
)

# 2. 챗 템플릿 조립
ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ROUTER_SYS_TEMPLATE, 
    ROUTER_HUMAN_TEMPLATE
])

def router_node(state: MCQState):
    """
    ==== 한국사 문제인지 / 아닌지 분기하는 노드 =====
    """
    llm = get_llm()
    
    # LCEL 체인 구성
    # 별도의 parser가 없다면 .content로 텍스트만 추출
    chain = ROUTER_PROMPT | llm
    
    print(f"🔍 [Router] 과목 분류 시작 (ID: {state['id']})")
    
    # 3. 실행
    response = chain.invoke({
        "paragraph": state["paragraph"], 
        "question": state["question"]
    })
    
    # ChatLlamaCpp의 결과는 AIMessage 객체이므로 .content 사용
    result = response.content.strip().upper()
    
    is_history = "KOREAN_HISTORY" in result
    
    print(f"📊 [Router] 분류 결과: {'한국사' if is_history else '일반'}")
    
    return {"is_korean_history": is_history, "retrieved_context": ""}

# 분기 결정 함수 (기존과 동일)
def route_decision(state: MCQState):
    if state.get("is_korean_history", False):
        return "retrieve"
    return "general_solve"