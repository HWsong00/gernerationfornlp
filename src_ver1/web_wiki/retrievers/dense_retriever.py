import gc
import torch
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

class DenseResourceManager:
    def __init__(self, model_name: str = "dragonkue/BGE-m3-ko"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

    def load_model(self):
        """임베딩 모델을 GPU에 상주시킵니다."""
        if self.model is None:
            print(f"🛰️ [Dense] 임베딩 모델 로드 중 ({self.model_name})...")
            self.model = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': self.device},
                encode_kwargs={'normalize_embeddings': True}
            )
        return self.model

    # 추론 시 상주시키므로 unload_model은 사용하지 않거나 비상용으로만 둡니다.
    def unload_model(self):
        """필요 시 GPU 메모리를 정리합니다."""
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()

def build_dense_retriever(splits: List[Document], resource_manager: DenseResourceManager):
    """
    전체 문서(Corpus)를 인덱싱하여 벡터 DB를 생성합니다.
    임베딩 모델이 상주된 상태에서 Chroma에 연결됩니다.
    """
    model = resource_manager.load_model()
    print(f"📦 [Dense] {len(splits)}개 청크 벡터 DB 생성 및 인덱싱 시작 (GPU)...")
    
    # 임베딩 모델 객체를 직접 전달하여 검색 시에도 자동 임베딩되도록 함
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=model,
        collection_name="history_dense_db"
    )
    return vectorstore

def get_dense_results(query: str, vectorstore: Chroma, top_k: int = 5):
    """
    추론 시점에 쿼리를 실시간 임베딩하여 유사 문서들을 검색합니다.
    """
    # vectorstore 생성 시 임베딩 모델이 연결되었으므로 텍스트만 전달하면 내부에서 GPU 연산 수행
    docs = vectorstore.similarity_search(query, k=top_k)
    return docs