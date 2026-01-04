from typing import List, Optional
from typing_extensions import TypedDict

class MCQState(TypedDict):
    """
    LangGraph 워크플로우를 위한 통합 상태 정의
    임베딩 및 30B LLM 추론 결과가 순차적으로 업데이트됩니다.
    """
    # 1. 원본 입력 데이터
    id: str                 # 문제 고유 ID
    paragraph: str          # 지문 원본
    question: str           # 질문 원본
    choices: List[str]      # 선지 리스트

    # 2. 분류 및 전략 (Classifier Node)
    is_korean_history: bool # 한국사 여부
    strategy: str           # 검색 전략 (INFERENCE / GENERAL) - 필요 시 활용

    # 3. 검색 및 컨텍스트 (Retriever Node)
    optimized_query: str    # LLM이 생성한 검색용 키워드/쿼리
    retrieved_context: str  # 앙상블 검색으로 확보한 외부 사료 데이터

    # 4. 추론 및 결과 (Solver & Parser Node)
    full_response: str      # LLM의 CoT 포함 전체 답변
    final_answer: str       # 최종 추출된 정답 번호 ("1", "2" 등)