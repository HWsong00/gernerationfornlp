import os
import json
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# 우리가 만든 모듈들 임포트
from utils.llm import get_llm
from utils.langfuse import langfuse_handler
from graph.workflow import create_mcq_workflow
from nodes.state import MCQState

# (주의) 이전 대화에서 정의한 ensemble_retriever 객체와 벡터 데이터 로드 필요
# 여기서는 이미 로드되어 있다고 가정하거나 초기화 함수를 호출합니다.
# from utils.retrieval_setup import init_ensemble_retriever 

def load_dataset(path):
    """869개의 문제가 담긴 JSON 데이터를 로드합니다."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main(is_test=False):
    load_dotenv()
    
    # 1. 환경 설정 및 데이터 로드
    DATA_PATH = os.getenv("DATA_PATH", "./data/history_test_set.json")
    OUTPUT_PATH = "./results/final_results.csv"
    os.makedirs("./results", exist_ok=True)
    
    dataset = load_dataset(DATA_PATH)
    if is_test:
        dataset = dataset[:3]  # 테스트 모드일 때는 3개만 실행
        print(f"🧪 테스트 모드 실행 (샘플 {len(dataset)}개)")

    # 2. 인프라 초기화 (Ensemble Retriever & Workflow)
    # ensemble_retriever는 장(Jang)님이 구성하신 BM25 + Chroma 객체입니다.
    # precomputed_vectors도 여기서 로드하여 상태에 주입합니다.
    print("🚀 인프라 초기화 중 (30B 모델 로드 포함)...")
    app = create_mcq_workflow(ensemble_retriever=None) # 여기에 실제 객체 주입
    
    results = []
    checkpoint_interval = 10 # 10문제마다 중간 저장

    # 3. 메인 루프 (tqdm으로 진행 상황 표시)
    print(f"🏃 전체 {len(dataset)}문항 풀이 시작!")
    
    for i, item in enumerate(tqdm(dataset)):
        problem_id = item.get('id', str(i))
        
        # 초기 상태 구성
        initial_state = {
            "id": problem_id,
            "paragraph": item['paragraph'],
            "question": item['question'],
            "choices": item['choices'],
            "precomputed_vectors": {}, # 사전 계산된 벡터 주입
            "is_korean_history": False,
            "final_answer": "N/A"
        }

        try:
            # 4. 워크플로우 실행 (Langfuse 핸들러 포함)
            config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}
            final_state = app.invoke(initial_state, config=config)
            
            # 결과 저장
            results.append({
                "id": problem_id,
                "answer": final_state.get("final_answer"),
                "full_log": final_state.get("full_response")
            })

        except Exception as e:
            print(f"❌ 에러 발생 (ID: {problem_id}): {e}")
            results.append({"id": problem_id, "answer": "ERROR", "full_log": str(e)})

        # 5. 중간 저장 (Checkpoint)
        if (i + 1) % checkpoint_interval == 0:
            pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
            # print(f"💾 중간 저장 완료 ({i+1}/{len(dataset)})")

    # 6. 최종 저장
    final_df = pd.DataFrame(results)
    final_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(f"🎉 모든 풀이 완료! 결과 저장: {OUTPUT_PATH}")

if __name__ == "__main__":
    # 실행 전 테스트: main(is_test=True)
    # 본 게임 실행: main(is_test=False)
    main(is_test=True)