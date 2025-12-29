"""
Parser Node: 모델 출력에서 정답 번호 추출
"""
from .state import MCQState
import re

def parser_node(state: MCQState):
    text = state['full_response']
    answer = None
    # JSON 형식 찾기
    match = re.search(r'{"정답":\s*"(\d+)"}', text)
    if match:
        answer = match.group(1)
    else:
        # 못 찾으면 텍스트 내 마지막 숫자 추출 
        nums = re.findall(r'\d+', text)
        if nums: answer = nums[-1]

    return {"final_answer": answer}
