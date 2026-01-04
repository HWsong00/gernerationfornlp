import re
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from nodes.state import MCQState
from utils.llm import get_llm

# 1. 키워드 추출용 템플릿 정의
KEYWORDS_GEN_SYS_TEMPLATE = SystemMessagePromptTemplate.from_template(
    """당신은 검색 최적화 전문가입니다. 제공된 문제의 내용을 분석하여 검색 엔진에서 관련 사료를 찾기 위한 핵심 키워드를 추출하세요.
    
    반드시 아래 형식을 지켜주세요:
    P: (지문의 핵심 사건, 인물, 유물 또는 핵심 문구 요약)
    Q: (질문에서 묻는 구체적인 대상이나 시기)
    C: (선지들에 공통적으로 등장하는 핵심 용어들)"""
)

def retrieve_node(state: MCQState, ensemble_retriever):
    """
    ==== 실시간 GPU 임베딩 기반 앙상블 검색 노드 ====
    """
    llm = get_llm()
    
    # --- Phase 1: 검색 최적화 키워드(optimized_query) 추출 ---
    kw_prompt = ChatPromptTemplate.from_messages([
        KEYWORDS_GEN_SYS_TEMPLATE,
        HumanMessagePromptTemplate.from_template("지문: {paragraph}\n질문: {question}\n선지: {choices}")
    ])
    
    kw_chain = kw_prompt | llm
    
    print(f"🛰️ [Retriever] 검색 쿼리 최적화 시작 (ID: {state['id']})")
    kw_result = kw_chain.invoke({
        "paragraph": state['paragraph'], 
        "question": state['question'], 
        "choices": "\n".join(state['choices'])
    }).content

    # --- Phase 2: 정규표현식 파싱 ---
    p_match = re.search(r"P:\s*(.*)", kw_result)
    q_match = re.search(r"Q:\s*(.*)", kw_result)
    c_match = re.search(r"C:\s*(.*)", kw_result)

    # --- Phase 3: 동적 검색 리스트 구성 (Multi-Query) ---
    search_tasks = [
        ("P_Raw", state['paragraph'][:200]), # 너무 길면 검색 노이즈가 생기므로 일부 절삭
        ("Q_Raw", state['question']),
    ]

    # 요약본(Summary) 추가
    if p_match and p_match.group(1).strip():
        search_tasks.append(("P_Sum", p_match.group(1).strip()))
    if q_match and q_match.group(1).strip():
        search_tasks.append(("Q_Sum", q_match.group(1).strip()))
    if c_match and c_match.group(1).strip():
        search_tasks.append(("C_Sum", c_match.group(1).strip()))

    # --- Phase 4: 앙상블 검색 실행 (실시간 GPU 임베딩) ---
    all_retrieved_docs = []
    
    for label, query in search_tasks:
        if not query.strip(): continue
        
        print(f"📡 [Retriever] {label} 검색 실행 중...")
        # [변경] p_vector 없이 텍스트 쿼리만 전달 -> ensemble.py에서 실시간 임베딩 수행
        docs = ensemble_retriever.invoke_ensemble(query)
        all_retrieved_docs.extend(docs)

    # --- Phase 5: 중복 제거 및 최종 컨텍스트 구성 ---
    unique_docs = []
    seen_contents = set()
    
    for d in all_retrieved_docs:
        if d.page_content not in seen_contents:
            unique_docs.append(d)
            seen_contents.add(d.page_content)

    # 상위 8개 문서 선택
    # 3개 문서로 수정
    final_docs = unique_docs[:3]
    context_str = "\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(final_docs)])

    print(f"✅ [Retriever] 검색 완료 (중복제거 후 {len(final_docs)}개 문서 확보)")

    # 업데이트된 상태 반환
    return {
        "optimized_query": kw_result, # 나중에 분석용으로 저장
        "retrieved_context": f"=== [교차 검증된 역사 사료 전문] ===\n{context_str}"
    }