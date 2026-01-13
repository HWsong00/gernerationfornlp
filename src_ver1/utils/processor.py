from typing import Optional
import pandas as pd
from tqdm import tqdm
from workflow.graph import create_rag_decision_workflow


def process(df, output_path: Optional[str] = None):
    """
    데이터프레임의 모든 문제 풀이

    Args:
        df: 문제 데이터프레임 (context, question_display, choices, id 컬럼 필요)
        output_path: 결과 저장 경로 (선택)

    Returns:
        results: 풀이 결과 리스트
    """
    app = create_rag_decision_workflow()
    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing problems"):
        context = row['context']
        question = row['question_display']
        choices = row['choices']
        row_id = row['id']

        initial_state = {
            "context": context,
            "question": question,
            "choices": choices,
            "row_id": row_id,
            "problem_type": "",
            "needs_external_knowledge": False,
            "rag_context": "",
            "reasoning": "",
            "final_answer": "",
            "confidence": 0.0
        }

        try:
            result = app.invoke(initial_state)
            results.append(result)

            # 진행 상황 출력
            print(f"\n[문제 {row_id}]")
            print(f"유형: {result['problem_type']}")
            print(f"RAG 사용: {result['needs_external_knowledge']}")
            print(f"판단 근거: {result['reasoning']}")
            print(f"신뢰도: {result['confidence']:.2f}")
            print(f"답변: {result['final_answer']}...")

            # 50 문제마다 중간 저장
            if len(results) % 50 == 0:
                results_df = pd.DataFrame(results)
                results_df.to_csv(output_path, index=False, encoding='utf-8-sig')

        except Exception as e:
            print(f"문제 {row_id} 처리 중 오류: {e}")
            results.append({
                **initial_state,
                "final_answer": f"오류: {str(e)}"
            })

    # 결과 저장
    if output_path:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n결과가 {output_path}에 저장되었습니다.")

    return results