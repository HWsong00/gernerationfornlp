import os
import gc
import torch
import time
import ast
import pandas as pd
from tqdm.auto import tqdm
from functools import partial

# 프로젝트 모듈 임포트
from config import settings
from data_load.loader import load_and_split_data
from retrievers.dense_retriever import DenseResourceManager, build_dense_retriever
from retrievers.sparse_retriever import LangChainKiwiBM25Retriever
from retrievers.ensemble import CustomWeightedEnsembleRetriever
from retrievers.tokenizer import tokenize_kiwi
from utils.llm import start_llama_server, get_llm_client  # 서버 시작 및 클라이언트 로직 포함
from utils.reranker import get_reranker
from utils.langfuse import langfuse_handler
from graph.workflow import create_mcq_workflow

def initialize_resources():
    """리소스 초기화 및 서버 가동 확인"""
    # 1. 서버 가동 및 대기 (20분 타임아웃 적용된 버전 호출)
    print("🌐 [0/3] LLM 서버 상태 확인 및 가동 중...")
    start_llama_server() 
    
    print("⚙️ [1/3] 리트리버 초기화 및 문서 인덱싱 중...")
    documents = load_and_split_data(settings.DATA_PATH)
    
    # Dense 리트리버 (BGE-M3)
    dense_manager = DenseResourceManager(model_name="dragonkue/BGE-m3-ko")
    vectorstore = build_dense_retriever(documents, dense_manager)
    
    # Sparse 리트리버 (Kiwi-BM25)
    corpus_tokenizer = partial(tokenize_kiwi, text_type="corpus")
    query_tokenizer = partial(tokenize_kiwi, text_type="query")
    sparse_retriever = LangChainKiwiBM25Retriever(
        documents=documents, k=10,
        corpus_tokenizer=corpus_tokenizer,
        query_tokenizer=query_tokenizer
    )
    
    # 앙상블 리트리버 조립
    ensemble_retriever = CustomWeightedEnsembleRetriever(
        sparse_retriever=sparse_retriever,
        vectorstore=vectorstore,
        weights=[0.3, 0.7], top_k=3
    )
    
    print("⚙️ [2/3] 리랭커 모델 로드 및 워크플로우 조립 중...")
    reranker_instance = get_reranker()
    # 람다 주입 방식으로 도구와 상태 분리
    app = create_mcq_workflow(ensemble_retriever, reranker_instance)
    
    print("✅ [3/3] 모든 리소스 및 서버 준비 완료")
    return app

def main():
    # 1. 시스템 초기화 (서버 가동 포함)
    try:
        app = initialize_resources()
    except Exception as e:
        print(f"🚨 초기화 중 치명적 오류 발생: {e}")
        return

    # 2. 데이터 로드 (경로는 본인 환경에 맞춰 수정)
    csv_path = "data/test.csv"
    if not os.path.exists(csv_path):
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {csv_path}")
        return
        
    train_df = pd.read_csv(csv_path)
    results = []
    
    print(f"🚀 총 {len(train_df)}문제 배치 풀이 시작 (Target: A100 80GB)")

    # 3. 배치 루프
    for index, row in tqdm(train_df.iterrows(), total=len(train_df)):
        try:
            # 문제 데이터 파싱
            prob_data = row['problems']
            if isinstance(prob_data, str):
                prob_data = ast.literal_eval(prob_data)
            if isinstance(prob_data, list):
                prob_data = prob_data[0]

            # 랭그래프 State 입력 데이터 (객체는 제외하고 순수 텍스트 데이터만!)
            sample_input = {
                "id": str(row['id']),
                "paragraph": row['paragraph'],
                "question": prob_data.get('question', ''),
                "choices": prob_data.get('choices', []),
            }

            # --- 워크플로우 실행 ---
            start_time = time.time()
            config = {
                "callbacks": [langfuse_handler], 
                "run_name": f"Batch_Run_{row['id']}"
            }

            # 랭퓨즈 타임아웃이나 모델 서버 일시적 오류 등으로 배치가 멈추지 않게 보호
            try:
                final_state = app.invoke(sample_input, config=config)
            except Exception as e:
                print(f"⚠️ 추론 실패 (ID: {row['id']}): {e}")
                # 실패한 데이터도 결과에는 남김 (추후 분석용)
                results.append({
                    "id": row['id'],
                    "is_correct": False,
                    "pred_answer": "ERROR",
                    "full_response": str(e)
                })
                continue

            latency = round(time.time() - start_time, 2)

            # 결과 수집 및 실시간 채점
            correct_ans = str(prob_data.get('answer', ''))
            pred_ans = str(final_state.get('final_answer', 'N/A'))
            
            results.append({
                "id": row['id'],
                "question": sample_input["question"],
                "correct_answer": correct_ans,
                "pred_answer": pred_ans,
                "is_correct": correct_ans == pred_ans,
                "full_response": final_state.get('full_response'),
                "latency": latency
            })

            # 4. 주기적 메모리 관리 (A100 VRAM 보호 루틴)
            # 5문제마다 캐시를 비워 리랭커와 서버의 VRAM 충돌 방지
            if index % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"🚨 심각한 데이터 에러 (Index: {index}): {e}")
            continue

    # 5. 결과 저장 및 통계
    results_df = pd.DataFrame(results)
    save_path = "result.csv"
    results_df.to_csv(save_path, index=False, encoding='utf-8-sig')

    if not results_df.empty:
        acc = (results_df['is_correct'].sum() / len(results_df)) * 100
        print(f"\n✨ 모든 배치 완료! 최종 정답률: {acc:.2f}%")
        print(f"💾 결과가 저장되었습니다: {save_path}")

if __name__ == "__main__":
    main()