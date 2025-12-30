import gc
from typing import List, Any, Optional, Callable
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field, PrivateAttr

class LangChainKiwiBM25Retriever(BaseRetriever):
    """
    저장 장치를 전혀 사용하지 않는 휘발성(In-Memory) Sparse 리트리버.
    인덱싱 직후 불필요한 객체를 파괴하여 RAM 효율을 극대화합니다.
    """
    k: int = Field(default=5)
    
    _docs: List[Document] = PrivateAttr()
    _bm25: Any = PrivateAttr()
    _query_tokenizer: Callable = PrivateAttr()

    def __init__(
        self,
        documents: List[Document],
        k: int,
        corpus_tokenizer: Callable,
        query_tokenizer: Callable,
        **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.k = k
        self._docs = documents
        self._query_tokenizer = query_tokenizer
        
        # 인덱싱 수행
        self._init_bm25(documents, corpus_tokenizer)

    def _init_bm25(self, documents: List[Document], tokenizer: Callable):
        """인덱싱 단계의 RAM 피크를 제어합니다."""
        print(f"⚡ [Sparse] {len(documents)}개 문서 기반 실시간 인덱싱 시작...")
        
        import bm25s
        
        # 1. 토큰화 (이 시점이 RAM을 가장 많이 먹는 순간)
        corpus_tokens = [tokenizer(doc.page_content) for doc in documents]
        
        # 2. 인덱스 생성
        self._bm25 = bm25s.BM25()
        self._bm25.index(corpus_tokens)
        
        # 3. [핵심] 인덱싱이 끝나자마자 토큰 리스트를 RAM에서 완전히 제거
        del corpus_tokens
        gc.collect() 
        
        print("✅ [Sparse] 인덱싱 완료 및 임시 메모리 반환.")

    def _get_relevant_documents(
        self, query: str, *, run_manager: Any = None
    ) -> List[Document]:
        # 쿼리 토크나이징 및 검색
        tokenized_query = [self._query_tokenizer(query)]
        results, scores = self._bm25.retrieve(tokenized_query, k=self.k)
        
        retrieved_docs = []
        for idx, score in zip(results[0], scores[0]):
            if score > 0:
                doc = self._docs[idx]
                # 원본 메타데이터에 스코어 기록 (필요 시 복사본 생성)
                doc.metadata["sparse_score"] = float(score)
                retrieved_docs.append(doc)
        
        return retrieved_docs