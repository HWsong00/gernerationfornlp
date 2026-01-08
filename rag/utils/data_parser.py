"""
데이터셋 파싱 유틸리티
"""

import pandas as pd
import ast


def parse_dataset(file_path):
    """
    CSV 파일에서 문제 데이터를 파싱하여 표준 형식으로 변환

    Args:
        file_path: CSV 파일 경로

    Returns:
        pd.DataFrame: 파싱된 문제 데이터프레임
            - id: 문제 ID
            - context: 지문 (paragraph)
            - question_display: 문제 + 보기 + 선택지 (포맷팅된 전체 문제)
            - choices: 선지 리스트
            - gt: 정답 (ground truth)
    """
    print(f"{file_path} 을 로딩 및 파싱합니다.")
    df = pd.read_csv(file_path)
    parsed_rows = []

    for idx, row in df.iterrows():
        try:
            # Flexible ID handling
            row_id = row.get('id', row.get('index', idx))
            context = row['paragraph'] if pd.notna(row['paragraph']) else ""

            # 문자열로 된 딕셔너리 파싱
            problem_data = ast.literal_eval(row['problems'])
            main_question = problem_data['question']
            choices = problem_data['choices']
            gt = problem_data['answer']

            q_plus = row.get('question_plus')
            if pd.notna(q_plus) and str(q_plus).strip():
                main_question += f"\n\n<보기>\n{q_plus}"

            formatted_choices = "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(choices)])
            full_question_display = f"{main_question}\n\n[선택지]\n{formatted_choices}"

            parsed_rows.append({
                "id": row_id,
                "context": context,
                "question_display": full_question_display,
                "choices": choices,
                "gt": gt
            })
        except Exception as e:
            print(f"오류 발생: {idx}: {e}")
            continue

    return pd.DataFrame(parsed_rows)