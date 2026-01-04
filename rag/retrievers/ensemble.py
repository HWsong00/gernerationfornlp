from typing import List, Any, Optional
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

class CustomWeightedEnsembleRetriever:
    def __init__(
        self, 
        sparse_retriever: Any, 
        vectorstore: Any, 
        weights: List[float] = [0.7, 0.3], 
        top_k: int = 4,
        k_constant: int = 20
    ):
        self.sparse_retriever = sparse_retriever
        self.vectorstore = vectorstore
        self.weights = weights
        self.top_k = top_k
        self.k_constant = k_constant

    def invoke_ensemble(self, query: str) -> List[Document]:
        """
        텍스트 쿼리를 받아 Sparse와 Dense 검색 결과를 실시간으로 통합합니다.
        (사전 계산된 벡터 대신 실시간 GPU 임베딩을 활용합니다.)
        """
        # [Step 1] Sparse 검색 (Kiwi-BM25)
        sparse_docs = self.sparse_retriever.invoke(query)
        
        # [Step 2] Dense 검색 (실시간 GPU 임베딩 기반)
        # dense_retriever.py에서 vectorstore 생성 시 임베딩 모델을 주입했으므로 
        # similarity_search만 호출해도 상주 중인 GPU 모델이 자동으로 벡터화하여 검색합니다.
        dense_docs = self.vectorstore.similarity_search(query, k=self.top_k)

        # [Step 3] RRF(Reciprocal Rank Fusion) 앙상블 계산
        all_docs = {}

        # Sparse 결과 처리
        for rank, doc in enumerate(sparse_docs):
            score = self.weights[0] / (self.k_constant + rank + 1)
            all_docs[doc.page_content] = {"doc": doc, "score": score}

        # Dense 결과 처리
        for rank, doc in enumerate(dense_docs):
            score = self.weights[1] / (self.k_constant + rank + 1)
            if doc.page_content in all_docs:
                all_docs[doc.page_content]["score"] += score
            else:
                all_docs[doc.page_content] = {"doc": doc, "score": score}

        # [Step 4] 최종 정렬 및 상위 k개 반환
        sorted_docs = sorted(all_docs.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs[:self.top_k]]