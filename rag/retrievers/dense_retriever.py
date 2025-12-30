import gc
import torch
import numpy as np
from typing import List, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

class DenseResourceManager:
    def __init__(self, model_name: str = "dragonkue/BGE-m3-ko"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

    def load_model(self):
        """임베딩 모델을 GPU에 로드합니다."""
        if self.model is None:
            print(f"🛰️ [Dense] 임베딩 모델 로드 중 ({self.model_name})...")
            self.model = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': self.device},
                encode_kwargs={'batch_size': 32} # 배치 사이즈 상향으로 속도 업
            )
        return self.model

    def unload_model(self):
        """GPU 메모리를 완전히 비웁니다."""
        if self.model is not None:
            print("🧹 [Dense] 임베딩 모델 제거 및 GPU 메모리 해제...")
            del self.model
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()

def build_dense_retriever(splits: List[Document], resource_manager: DenseResourceManager):
    """전체 문서(Corpus)를 인덱싱하여 벡터 DB를 생성합니다."""
    model = resource_manager.load_model()
    print(f"📦 [Dense] {len(splits)}개 청크 벡터 DB 생성 시작 (GPU)...")
    
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=model,
        collection_name="history_dense_db"
    )
    return vectorstore

def pre_embed_eval_dataset(evaluation_data: List[Dict], resource_manager: DenseResourceManager):
    """
    869개 테스트셋의 모든 텍스트를 GPU 배치로 사전 임베딩합니다.
    결과는 {문제_ID: {타입: 벡터}} 형태로 반환하여 나중에 CPU 연산 없이 검색하게 합니다.
    """
    model = resource_manager.load_model()
    
    # 1. 모든 텍스트 추출 (중복 제거를 위해 set 사용 고려 가능)
    all_texts = []
    text_mapping = [] # (problem_id, type, index) 보관용
    
    for item in evaluation_data:
        p_id = item.get('id', 'unknown')
        # 지문, 질문, 선지들 순서대로 추가
        texts_to_embed = [
            ("paragraph", item['paragraph']),
            ("question", item['question']),
        ] + [("choice", c) for c in item['choices']]
        
        for t_type, content in texts_to_embed:
            all_texts.append(content)
            text_mapping.append((p_id, t_type))

    print(f"🚀 [Dense] 총 {len(all_texts)}개 테스트 텍스트 배치 임베딩 시작...")
    
    # 2. GPU 배치 임베딩 실행
    vectors = model.embed_documents(all_texts)
    
    # 3. 구조화된 형태로 재조립
    pre_computed_vectors = {}
    for (p_id, t_type), vector in zip(text_mapping, vectors):
        if p_id not in pre_computed_vectors:
            pre_computed_vectors[p_id] = {}
        
        # 선지는 여러 개이므로 리스트로 관리
        if t_type == "choice":
            if "choices" not in pre_computed_vectors[p_id]:
                pre_computed_vectors[p_id]["choices"] = []
            pre_computed_vectors[p_id]["choices"].append(vector)
        else:
            pre_computed_vectors[p_id][t_type] = vector
            
    print("✅ [Dense] 테스트셋 사전 임베딩 완료.")
    return pre_computed_vectors