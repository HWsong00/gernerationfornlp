from typing import List, Any, Optional
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

class CustomWeightedEnsembleRetriever:
    def __init__(
        self, 
        sparse_retriever: Any, 
        vectorstore: Any, # 리트리버 객체 대신 vectorstore를 직접 받음
        weights: List[float] = [0.7, 0.3], 
        top_k: int = 4,
        k_constant: int = 20
    ):
        self.sparse_retriever = sparse_retriever
        self.vectorstore = vectorstore
        self.weights = weights
        self.top_k = top_k
        self.k_constant = k_constant

    def invoke_ensemble(self, query: str, query_vector: List[float]) -> List[Document]:
        """
        텍스트 쿼리와 사전 계산된 벡터를 동시에 사용하여 앙상블 검색을 수행합니다.
        """
        # [Step 1] Sparse 검색 (텍스트 기반)
        sparse_docs = self.sparse_retriever.invoke(query)
        
        # [Step 2] Dense 검색 (사전 계산된 벡터 기반)
        # 임베딩 모델을 거치지 않고 바로 벡터로 검색하므로 VRAM 사용량 0, 속도 최상!
        dense_docs = self.vectorstore.similarity_search_by_vector(query_vector, k=self.top_k)

        # [Step 3] RRF(Reciprocal Rank Fusion) 앙상블 계산
        all_docs = {}

        # Sparse 결과 처리 (BM25)
        for rank, doc in enumerate(sparse_docs):
            score = self.weights[0] / (self.k_constant + rank + 1)
            all_docs[doc.page_content] = {"doc": doc, "score": score}

        # Dense 결과 처리 (Vector)
        for rank, doc in enumerate(dense_docs):
            score = self.weights[1] / (self.k_constant + rank + 1)
            if doc.page_content in all_docs:
                all_docs[doc.page_content]["score"] += score
            else:
                all_docs[doc.page_content] = {"doc": doc, "score": score}

        # [Step 4] 최종 정렬 및 상위 k개 반환
        sorted_docs = sorted(all_docs.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs[:self.top_k]]