"""
MCQState: LangGraph에서 사용할 상태(State) 정의
"""
from typing import List
from typing_extensions import TypedDict

class MCQState(TypedDict):
    """문제 풀이를 위한 상태"""
    id: str
    paragraph: str
    question: str
    choices: List[str]
    is_history: bool       # 한국사 문제 여부
    strategy: str         # INFERENCE vs GENERAL
    summary: str          # 지문 요약본
    optimized_query: str  # 생성된 키워드 (10개 이내)
    retrieved_context: str
    full_response: str
    final_answer: str