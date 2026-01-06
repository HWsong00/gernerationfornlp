import json
from nodes.state import MCQState
from utils.llm import get_llm_client, MODEL_NAME
from utils.wiki import WikipediaAPI, WikiChunker 

def retrieve_node(state: MCQState, ensemble_retriever, reranker):
    """
    ==== 고도화된 지식 검색 노드 (Native OpenAI SDK / No Truncation) ====
    """
    client = get_llm_client()
    
    # --- Phase 1: 검색 키워드 추출 (Native JSON Mode) ---
    system_content = (
        "당신은 검색 최적화 전문가입니다. 제공된 문제 내용을 분석하여 위키피디아 검색에 최적화된 핵심 용어 3개를 추출하세요.\n"
        "반드시 ['키워드1', '키워드2', '키워드3'] 형식의 JSON 리스트로만 답변하십시오."
    )
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"지문: {state.get('paragraph', '')}\n질문: {state.get('question', '')}"}
    ]

    print(f"🔑 [Retriever] 키워드 추출 시작 (ID: {state.get('id', 'unknown')})")
    
    try:
        kw_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=150  # 키워드가 길어질 수 있으므로 여유 있게 설정
        )
        
        raw_content = kw_response.choices[0].message.content.strip()
        parsed_json = json.loads(raw_content)
        
        # JSON 구조에 따른 유연한 파싱
        if isinstance(parsed_json, list):
            search_queries = parsed_json
        elif isinstance(parsed_json, dict) and 'keywords' in parsed_json:
            search_queries = parsed_json['keywords']
        else:
            search_queries = list(parsed_json.values())[0] if parsed_json else []
            
        if not search_queries: raise ValueError("Empty keywords")
        
    except Exception as e:
        # [수정] 너무 길면 자르는 로직 제거: 원본 질문 전체를 쿼리로 사용
        print(f"⚠️ 키워드 추출 실패({e}), 질문 전체를 검색 쿼리로 사용합니다.")
        search_queries = [state.get('question', '')]

    print(f"🔎 [Retriever] 최종 쿼리 리스트: {search_queries}")

    # --- Phase 2: 다중 출처 검색 (Local + Wiki) ---
    candidate_docs = []
    
    # 1. 로컬 앙상블 검색
    for query in search_queries[:2]:
        candidate_docs.extend(ensemble_retriever.invoke_ensemble(query))

    # 2. 위키백과 검색
    try:
        wiki_api = WikipediaAPI()
        chunker = WikiChunker()
        wiki_raw = wiki_api.search_and_fetch(search_queries)
        wiki_chunks = chunker.chunk(wiki_raw)
        for ch in wiki_chunks[:10]:
            candidate_docs.append(ch['text']) 
    except Exception as e:
        print(f"❌ 위키 검색 실패: {e}")

    # 중복 제거 및 텍스트 추출
    raw_texts = []
    for d in candidate_docs:
        text = d.page_content if hasattr(d, 'page_content') else d
        if text not in raw_texts:
            raw_texts.append(text)

    # --- Phase 3: 리랭킹 (Reranking) ---
    print(f"⚖️ [Reranker] {len(raw_texts)}개 문서 재정렬 시작...")
    
    combined_query = f"{state['question']} {' '.join(state['choices'])}"
    reranked_results = reranker.rerank(combined_query, raw_texts, top_k=3)
    
    final_context = []
    for i, (text, score) in enumerate(reranked_results):
        final_context.append(f"[{i+1}] (신뢰도: {score:.2f}) {text}")

    context_str = "\n\n".join(final_context)
    print(f"✅ [Retriever] 최종 컨텍스트 구성 완료")

    return {
        "retrieved_context": f"=== [엄선된 지식 컨텍스트] ===\n{context_str}"
    }