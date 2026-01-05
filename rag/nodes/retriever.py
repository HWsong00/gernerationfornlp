import json
import re
from nodes.state import MCQState
from utils.llm import get_llm
from utils.wiki import WikipediaAPI, WikiChunker # 이제 임포트 가능!

def retrieve_node(state: MCQState, ensemble_retriever):
    """
    ==== 고도화된 지식 검색 노드 (needs_knowledge 기반) ====
    """
    llm = get_llm()
    
    # --- Phase 1: 검색 키워드 추출 (OpenAI Format) ---
    system_content = (
        "당신은 검색 최적화 전문가입니다. 제공된 문제 내용을 분석하여 검색 엔진에서 관련 정보를 찾기 위한 핵심 키워드를 추출하세요.\n\n"
        "반드시 아래 형식을 지켜주세요:\n"
        "P: (지문의 핵심 사건, 인물 요약)\n"
        "Q: (질문에서 묻는 핵심 대상)\n"
        "C: (선지들의 공통 핵심 용어)"
    )
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"지문: {state.get('paragraph', '')}\n질문: {state.get('question', '')}"}
    ]

    try:
        kw_response = llm.invoke(messages).content
        search_queries = [m.group(1).strip() for m in re.finditer(r"[PQC]:\s*(.*)", kw_response)]
        if not search_queries: raise ValueError()
    except:
        # 실패 시 질문 전체 검색 (장님 요청사항)
        print("⚠️ 키워드 추출 실패로 질문 전체를 사용합니다.")
        search_queries = [state.get('question', '')]

    # --- Phase 2: 다중 출처 검색 (Local + Wiki) ---
    candidate_docs = []
    
    # 1. 로컬 앙상블 (쿼리당 5개씩 넉넉히 확보)
    for query in search_queries[:2]:
        candidate_docs.extend(ensemble_retriever.invoke_ensemble(query))

    # 2. 위키백과 (넉넉히 확보)
    try:
        wiki_api = WikipediaAPI()
        chunker = WikiChunker()
        wiki_raw = wiki_api.search_and_fetch(search_queries)
        wiki_chunks = chunker.chunk(wiki_raw)
        for ch in wiki_chunks[:10]:
            candidate_docs.append(ch['text']) # 텍스트 형태로 저장
    except Exception as e:
        print(f"❌ 위키 검색 실패: {e}")

    # Document 객체에서 텍스트만 추출 및 중복 제거
    raw_texts = []
    for d in candidate_docs:
        text = d.page_content if hasattr(d, 'page_content') else d
        if text not in raw_texts:
            raw_texts.append(text)

    # --- Phase 3: 리랭킹 (Reranking) ---
    print(f"⚖️ [Reranker] {len(raw_texts)}개 문서 재정렬 시작...")
    
    # 질문과 선지를 합쳐서 쿼리로 사용 (리랭킹 정확도 향상)
    combined_query = f"{state['question']} {' '.join(state['choices'])}"
    
    # 리랭커 실행 (상위 3개 선발)
    reranked_results = reranker.rerank(combined_query, raw_texts, top_k=3)
    
    # 최종 컨텍스트 구성
    final_context = []
    for i, (text, score) in enumerate(reranked_results):
        final_context.append(f"[{i+1}] (신뢰도: {score:.2f}) {text}")

    context_str = "\n\n".join(final_context)

    print(f"✅ [Retriever] 리랭킹 완료 (최종 3개 문서 선발)")

    return {
        "retrieved_context": f"=== [엄선된 지식 컨텍스트] ===\n{context_str}"
    }