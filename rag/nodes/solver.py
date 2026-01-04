import re
import json
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from nodes.state import MCQState
from utils.llm import get_llm

# =========================================================
# 1. 한국사 전용 Solver Node (RAG + CoT Guard)
# =========================================================
def ko_history_solver_node(state: MCQState):
    """
    한국사 전문가 페르소나를 사용하여 검색된 사료(retrieved_context)를 기반으로 추론합니다.
    """
    llm = get_llm()
    
    system_template = SystemMessagePromptTemplate.from_template(
        """당신은 사료 해석에 능숙한 한국사 전문가입니다. 
제공된 <개념 보충 자료>의 각 문서([1], [2] 등)를 근거로 문제를 해결하십시오.

** [추론 루프 방지 및 불능 문제 대응 규칙] **
1. **단계적 분석(<think>)**: 답변 전 <think> 태그 내에서 최대 7단계까지만 추론하십시오.
2. **무한 루프 금지**: 7단계 내에 결론이 나지 않으면 즉시 추론을 멈추고 가장 가능성 높은 답을 선택하십시오.
3. **최종 형식**: 반드시 마지막 줄에 {{"정답": "번호"}} 형식으로 답을 출력하십시오."""
    )

    human_template = HumanMessagePromptTemplate.from_template(
        """<지문>
{paragraph}

<개념 보충 자료>
{retrieved_context}

<질문>
{question}

<선지>
{choices}"""
    )

    prompt = ChatPromptTemplate.from_messages([system_template, human_template])
    
    # 선지 포맷팅
    choices_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(state['choices'])])

    print(f"🤖 [History Solver] 추론 시작 (ID: {state['id']})")
    
    chain = prompt | llm
    response = chain.invoke({
        "paragraph": state['paragraph'],
        "question": state['question'],
        "choices": choices_str,
        "retrieved_context": state.get('retrieved_context', "관련 사료를 찾지 못했습니다.")
    })

    return {"full_response": response.content}


# =========================================================
# 2. 일반 과목 Solver Node (Pure CoT Guard)
# =========================================================
def general_solver_node(state: MCQState):
    """
    한국사 외 과목을 위한 노드입니다. 외부 자료 없이 지문의 논리 구조에 집중합니다.
    """
    llm = get_llm()
    
    system_template = SystemMessagePromptTemplate.from_template(
        """당신은 논리적이고 객관적인 수험생입니다. 주어진 지문만을 분석하여 정답을 고르십시오.

** [추론 및 종료 규칙] **
1. **단계적 사고(<think>)**: <think> 태그 내에서 최대 7단계의 논리 전개를 수행하십시오.
2. **최선책 선택**: 정보가 부족하더라도 가장 타당해 보이는 번호를 고르십시오.
3. **형식 준수**: 마지막 줄은 반드시 {{"정답": "번호"}} 입니다."""
    )

    human_template = HumanMessagePromptTemplate.from_template(
        """[지문]: {paragraph}\n[질문]: {question}\n[선지]:\n{choices}"""
    )

    prompt = ChatPromptTemplate.from_messages([system_template, human_template])
    choices_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(state['choices'])])

    print(f"🤖 [General Solver] 추론 시작 (ID: {state['id']})")

    chain = prompt | llm
    response = chain.invoke({
        "paragraph": state['paragraph'],
        "question": state['question'],
        "choices": choices_str
    })

    return {'full_response': response.content}


# =========================================================
# 3. Recovery Node (비상 정답 추출 노드)
# =========================================================
def recovery_node(state: MCQState):
    """
    Solver가 형식을 지키지 못했을 때, 추론 로그에서 정답 번호를 강제로 추출합니다.
    """
    llm = get_llm()
    
    recovery_prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 채점관입니다. 아래 추론 내용에서 최종 정답 번호 하나만 숫자만 출력하십시오."),
        ("human", "이전 추론 내용: {full_response}\n\n결국 정답은 몇 번입니까?")
    ])
    
    print(f"🚨 [Recovery] 비상 정답 추출 시도 (ID: {state['id']})")
    
    # [수정] invoke 시 필요한 변수(full_response)를 전달해야 합니다.
    chain = recovery_prompt | llm
    response = chain.invoke({"full_response": state['full_response']})
    
    # 숫자 하나만 추출
    match = re.search(r"\d", response.content)
    final_answer = match.group(0) if match else "1" 
    
    return {"final_answer": final_answer}