import gc
import torch
import json
from nodes.state import MCQState
from utils.llm import get_llm_client, MODEL_NAME
from utils.wiki import WikipediaAPI, WikiChunker
from utils.reranker import Reranker

def retrieve_node(state: MCQState, ensemble_retriever, reranker):
    client = get_llm_client()
    
    # [수정] state에서 꺼내지 말고, 인자로 받은 객체를 그대로 사용합니다.
    # 변수명을 기존 코드와 맞춰주면 아래 로직을 고칠 필요가 없어 편리합니다.
    reranker_obj = reranker
    retriever_obj = ensemble_retriever

    # --- Phase 1: 검색 키워드 추출 (Robust Version) ---
    system_content = (
        "당신은 검색 전문가입니다. 반드시 {\"keywords\": [\"용어1\", \"용어2\"]} 형식의 JSON으로만 답변하십시오."
    )
    messages = [{"role": "system", "content": system_content},
                {"role": "user", "content": f"지문: {state.get('paragraph', '')}\n질문: {state.get('question', '')}"}]

    try:
        kw_response = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, temperature=0,
            response_format={"type": "json_object"}
        )
        parsed_json = json.loads(kw_response.choices[0].message.content)
        
        # 어떤 구조로 오든 무조건 List[str] 보장
        raw_val = parsed_json.get('keywords', list(parsed_json.values())[0] if parsed_json else [])
        search_queries = raw_val if isinstance(raw_val, list) else [str(raw_val)]
        search_queries = [q.strip() for q in search_queries if len(q.strip()) > 1][:3]
    except Exception as e:
        print(f"⚠️ 키워드 추출 실패: {e}")
        search_queries = [state.get('question', '')[:20]]

    print(f"🔎 [Retriever] 최종 쿼리 리스트: {search_queries}")

    # --- Phase 2: 다중 출처 검색 (Efficiency Optimized) ---
    candidate_docs = []
    # [수정] API 객체는 루프 밖에서 한 번만 생성
    wiki_api = WikipediaAPI()
    chunker = WikiChunker()

    for q in search_queries:
        candidate_docs.extend(retriever_obj.invoke_ensemble(q))
        try:
            wiki_raw = wiki_api.search_and_fetch([q])
            if wiki_raw:
                # [수정] 위키 문서가 너무 많아지지 않게 조절
                wiki_chunks = chunker.chunk(wiki_raw)
                for ch in wiki_chunks[:5]: # 쿼리당 위키 청크는 5개로 제한
                    candidate_docs.append(ch['text'])
        except: pass

    # 중복 제거 및 [핵심] 길이 제한(Truncation)
    raw_texts = []
    seen = set()
    for d in candidate_docs:
        text = d.page_content if hasattr(d, 'page_content') else str(d)
        if text not in seen:
            # [수정] 리랭커 VRAM 보호를 위해 문서당 1000자로 제한
            raw_texts.append(text[:1000]) 
            seen.add(text)

    # --- Phase 3: 리랭킹 및 메모리 정리 ---
    combined_query = f"{state['question']} {' '.join(state['choices'])}"
    print(f"⚖️ [Reranker] {len(raw_texts)}개 문서 재정렬 시작")
    
    reranked_results = reranker_obj.rerank(combined_query, raw_texts, top_k=3)
    final_context = [f"[{i+1}] (신뢰도: {score:.2f}) {text}" for i, (text, score) in enumerate(reranked_results)]

    # 메모리 정리
    del candidate_docs, raw_texts
    gc.collect()
    torch.cuda.empty_cache()

    # 이 부분이 최종입니다. 더 이상 수정 안 하셔도 됩니다!
    return {
        "retrieved_context": f"=== [엄선된 지식 컨텍스트] ===\n" + "\n\n".join(final_context),
        "optimized_query": ", ".join(search_queries) # 키 이름을 State와 일치시킴
    }