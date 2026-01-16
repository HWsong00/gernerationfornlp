from sentence_transformers import CrossEncoder
from functools import lru_cache
import torch
from typing import List, Dict

RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        # GPU 가용 여부에 따라 장치 설정
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=self.device)

    def rerank(self, query: str, docs: List[str], top_k: int = 5):
        """
        query와 문자열 리스트(docs)를 받아 점수순으로 정렬하여 반환합니다.
        """
        if not docs:
            return []

        # [Query, Doc] 쌍 생성
        pairs = [[query, d] for d in docs]
        
        # Cross-Encoder 추론
        with torch.inference_mode():
            scores = self.model.predict(pairs)
        
        # 문서와 점수 매칭 및 정렬
        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        
        return scored_docs[:top_k]

@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker()