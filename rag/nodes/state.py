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


    #######################################
    from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

class MCQState(TypedDict):
    """
    LangGraph 워크플로우를 위한 통합 상태 정의
    """
    # 1. 기본 입력 데이터
    id: str                 # 문제 고유 ID
    paragraph: str          # 원본 지문
    question: str           # 질문 내용
    choices: List[str]      # 선지 리스트 (4~5개)

    # 2. 분류 및 전략 (Router & Retriever)
    is_korean_history: bool # 한국사 여부 (router_node 결과)
    strategy: str           # 검색 전략 (INFERENCE / GENERAL)
    
    # 3. 인프라 및 검색 (Retriever)
    # RAM에 저장된 사전 계산 벡터 맵 ({'id': {'paragraph': [...], ...}})
    precomputed_vectors: Dict[str, Any] 
    retrieved_context: str  # 최종 앙상블 검색 결과 (8개 문서 통합본)

    # 4. 추론 및 결과 (Solver & Parser)
    full_response: str      # LLM의 전체 답변 ( <think> 포함 )
    final_answer: str       # 최종 추출된 정답 번호 ("1", "2", "3" 등)

    # (선택 사항) 로깅 및 분석용
    # kw_result: str        # LLM이 생성한 P/Q/C 키워드 원문 (필요 시)