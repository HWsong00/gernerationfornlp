###################################################################################

import re
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from nodes.state import MCQState
from utils.llm import get_llm

def retrieve_node(state: MCQState, ensemble_retriever):
    llm = get_llm()
    p_id = state['id']
    
    # --- Phase 1: 키워드 추출 ---
    kw_prompt = ChatPromptTemplate.from_messages([
        KEYWORDS_GEN_SYS_TEMPLATE,
        HumanMessagePromptTemplate.from_template("지문: {paragraph}\n질문: {question}\n선지: {choices}")
    ])
    kw_result = kw_chain.invoke({
        "paragraph": state['paragraph'], "question": state['question'], "choices": "\n".join(state['choices'])
    }).content

    # --- Phase 2: 정규표현식 파싱 (None 가능성 유지) ---
    p_match = re.search(r"P:\s*(.*)", kw_result)
    q_match = re.search(r"Q:\s*(.*)", kw_result)
    c_match = re.search(r"C:\s*(.*)", kw_result)

    # --- Phase 3: 동적 검색 리스트 구성 ---
    # 원본(Raw) 검색은 무조건 수행
    search_tasks = [
        ("P_Raw", state['paragraph']),
        ("Q_Raw", state['question']),
        ("C_Raw", " ".join(state['choices']))
    ]

    # 파싱에 성공한 요약본(Summary)이 있을 때만 검색 리스트에 추가
    if p_match and p_match.group(1).strip():
        search_tasks.append(("P_Sum", p_match.group(1).strip()))
    if q_match and q_match.group(1).strip():
        search_tasks.append(("Q_Sum", q_match.group(1).strip()))
    if c_match and c_match.group(1).strip():
        search_tasks.append(("C_Sum", c_match.group(1).strip()))

    # --- Phase 4: 앙상블 검색 실행 ---
    all_retrieved_docs = []
    p_vector = state['precomputed_vectors'][p_id]['paragraph']

    for label, query in search_tasks:
        print(f"📡 [Retriever] {label} 검색 실행...")
        docs = ensemble_retriever.invoke_ensemble(query, p_vector)
        all_retrieved_docs.extend(docs)

    # --- Phase 5: 중복 제거 및 최종 컨텍스트 구성 ---
    unique_docs = []
    seen = set()
    for d in all_retrieved_docs:
        if d.page_content not in seen:
            unique_docs.append(d)
            seen.add(d.page_content)

    final_docs = unique_docs[:8]
    context_str = "\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(final_docs)])

    return {"retrieved_context": f"=== [교차 검증된 역사 사료 전문] ===\n{context_str}"}