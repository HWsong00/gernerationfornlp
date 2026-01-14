#############################################
import re
from nodes.state import MCQState

def parser_node(state: MCQState):
    """
    LLM의 답변(full_response)에서 최종 정답 번호만 추출하는 노드
    """
    text = state.get('full_response', "")
    answer = None
    
    print(f"🎯 [Parser] 정답 추출 시도 (ID: {state['id']})")

    # 1. 표준 JSON 형식 찾기 
    # {"정답": "1"} 또는 {"정답": 1} 또는 {'정답': '1'} 등 다양한 따옴표/공백 대응
    # r'\{["\']정답["\']:\s*["\']?(\d)["\']?\}'
    json_match = re.search(r'\{["\']정답["\']:\s*["\']?(\d)["\']?\}', text)
    
    if json_match:
        answer = json_match.group(1)
        print(f"✅ [Parser] JSON 형식에서 추출 성공: {answer}")
    else:
        # 2. JSON 형식이 없을 경우 최후의 수단: 텍스트 내 마지막 숫자 추출
        # 보통 모델이 결론을 마지막에 내리므로 findall의 마지막 인덱스를 가져옵니다.
        nums = re.findall(r'\d', text)
        if nums:
            answer = nums[-1]
            print(f"⚠️ [Parser] JSON 실패, 텍스트 내 마지막 숫자 추출: {answer}")
        else:
            print(f"❌ [Parser] 정답 추출 실패 (Recovery 노드로 이동 예정)")
            answer = "N/A" # 확실하게 실패를 알림

    return {"final_answer": answer}