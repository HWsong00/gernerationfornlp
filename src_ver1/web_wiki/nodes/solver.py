import re
import json
from nodes.state import MCQState
from utils.llm import get_llm_client, MODEL_NAME

# =========================================================
# 1. 한국사 전용 Solver Node (Native SDK + RAG)
# =========================================================
def ko_history_solver_node(state: MCQState):
    """
    한국사 전문가 페르소나를 사용하여 Native SDK로 추론합니다.
    (랭체인 걷어내기: 템플릿 대신 Raw 메시지 사용)
    """
    client = get_llm_client()
    
    # 선지 포맷팅
    choices_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(state['choices'])])
    
    # 팀원의 성공 비결: 강력한 시스템 메시지와 제약 조건
    system_msg = """당신은 사료 해석에 능숙한 한국사 전문가입니다. 
제공된 <개념 보충 자료>의 각 문서([1], [2] 등)를 근거로 문제를 해결하십시오.

[절대 규칙]
1. 답변 전 <think> 태그 내에서 단계별로 추론하되, 했던 말을 절대 반복하지 마십시오.
2. 7단계 내에 결론이 나지 않으면 즉시 추론을 멈추고 가장 가능성 높은 답을 선택하십시오.
3. 정답은 반드시 선택지 번호 중 하나여야 하며, 마지막에 JSON 형식 {"정답": "번호"}로 답변을 종료하십시오."""

    user_msg = f"""<지문>
{state['paragraph']}

<개념 보충 자료>
{state.get('retrieved_context', "관련 자료 없음")}

<질문>
{state['question']}

<선지>
{choices_str}"""

    print(f"🤖 [History Solver] Native 추론 시작 (ID: {state['id']})")
    
    # Native 호출: repetition_penalty 주입으로 루프 물리적 차단
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.7,  # 낮은 온도로 일관성 유지
        max_tokens=8192,
        extra_body={
            "repetition_penalty": 1.15, # 루프를 방지하는 가장 강력한 장치
        },
        stop=["<|im_end|>", "###"] # 답변이 늘어지는 것을 방지
    )
    
    return {"full_response": response.choices[0].message.content}


# =========================================================
# 2. 일반 과목 Solver Node (Native SDK)
# =========================================================
def general_solver_node(state: MCQState):
    """
    일반 과목을 위한 Native SDK 노드입니다.
    """
    client = get_llm_client()
    
    choices_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(state['choices'])])
    
    system_msg = """당신은 논리적이고 객관적인 수험생입니다. 주어진 지문만을 분석하여 정답을 고르십시오.

[절대 규칙]
1. <think> 태그 내에서 최대 7단계의 논리 전개를 수행하십시오. 똑같은 문장을 반복하지 마십시오.
2. 정보가 부족하더라도 가장 타당해 보이는 번호를 고르십시오.
3. 마지막 줄은 반드시 {"정답": "번호"} 형식이어야 합니다."""

    user_msg = f"[지문]: {state['paragraph']}\n[질문]: {state['question']}\n[선지]:\n{choices_str}"

    print(f"🤖 [General Solver] Native 추론 시작 (ID: {state['id']})")

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.7,
        max_tokens=8192,
        extra_body={
            "repetition_penalty": 1.15
        }
    )

    return {'full_response': response.choices[0].message.content}


# =========================================================
# 3. Recovery Node (Native SDK 기반 비상 추출)
# =========================================================
def recovery_node(state: MCQState):
    """
    Native SDK를 사용하여 추론 로그에서 정답 번호를 강제로 추출합니다.
    """
    client = get_llm_client()
    
    messages = [
        {"role": "system", "content": "당신은 채점관입니다. 아래 추론 내용에서 최종 정답 번호 하나만 숫자만 출력하십시오. 다른 말은 하지 마세요."},
        {"role": "human", "content": f"이전 추론 내용: {state['full_response']}\n\n결국 정답은 몇 번입니까?"}
    ]
    
    print(f"🚨 [Recovery] Native 비상 정답 추출 (ID: {state['id']})")
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0,
        max_tokens=10 # 숫자만 필요하므로 최소화
    )
    
    content = response.choices[0].message.content
    match = re.search(r"\d", content)
    final_answer = match.group(0) if match else "1" 
    
    return {"final_answer": final_answer}