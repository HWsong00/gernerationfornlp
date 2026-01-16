from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict

class MCQState(TypedDict):
    """
    개선된 MCQState: 데이터와 도구를 분리했습니다.
    """
    # 1. 원본 입력 데이터
    id: str
    paragraph: str
    question: str
    choices: List[str]
    # [제거] reranker, ensemble_retriever는 여기서 빠집니다.

    # 2. 분류 및 전략
    needs_knowledge: bool  # 지식 기반 검색 필요 여부
    category: str          # 과목 분류 (한국사, 일반 등)

    # 3. 검색 및 컨텍스트
    # 쿼리가 여러 개일 수 있으므로 List[str] 추천
    optimized_queries: List[str] 
    retrieved_context: str  # 정제된 텍스트 컨텍스트

    # 4. 추론 및 결과
    full_response: str      # LLM CoT 답변
    final_answer: str       # 최종 정답 ("1")