import pandas as pd

def merge_c_r(chosen_path, rejected_path):
    chosen = pd.read_csv(chosen_path)
    rejected = pd.read_csv(rejected_path)

    merged = pd.merge(chosen, rejected[['id', 'full', 'model_answer']], on='id', how='inner')
    return merged
    

def create_prompt(row):
    context = row['input_context']
    question = row['input_question']
    
    return f"""
너는 대한민국 최고의 명문대학교에서 수능형 문항 분석과 교수법을 담당하는 교육학 교수이다.
너의 목표는 학생들에게 복잡한 수능형 지문과 문제를 어떻게 논리적으로 풀고 분석해야 하는지 가르치는 것이다.
단순히 정답만 맞히는 것이 아니라, 지문에 근거하여 오답이 왜 오답인지까지 명확하게 설명하는 '가장 완벽한 해설'을 제공해야 한다.

<task_description>
아래에 수능형 문제의 평가 가이드라인과 실제 문항(지문 및 발문)이 제시될 것이다. 
너는 이 가이드라인을 준수하여 해당 문제에 대한 정답과 심층적인 풀이 과정을 작성해야 한다.
</task_description>

<guideline>
수능형 문제의 풀이는 다음의 세 가지 관점에서 평가된다:
- (Logical Grounding, 논리적 근거): 모든 풀이는 반드시 지문 내의 명시적 단서에 근거해야 한다.
- (Conceptual Accuracy, 개념적 정확성): 국어, 사회, 역사 등 각 과목의 핵심 개념과 용어를 정확하게 사용해야 한다. 특히 인과관계나 사회적 현상의 정의가 틀림이 없어야 한다.
- (Process Transparency, 풀이 과정의 투명성): 정답이 도출되는 단계를 1) 지문 분석, 2) 선택지별 분석(정답 및 오답 이유), 3) 최종 결론의 순서로 명확히 제시하여 학생이 사고의 흐름을 따라올 수 있게 해야 한다.
</guideline>

아래는 네가 분석해야 할 문항이다.
<context_and_question>
[지문]
{context}

[질문]
{question}
</context_and_question>

지문을 꼼꼼히 읽고 위의 가이드라인을 복습하라. 
작성된 해설은 반드시 <response></response> 태그 안에 넣어라.
해설이 끝난 후, 반드시 맨 마지막 줄에 최종 정답을 {{"정답": n}} 형식의 문자열을 추가하라.
""".strip()

def add_category_split(merged, category_path, train_path, valid_path):
    merged['prompt'] = merged.apply(create_prompt, axis=1)

    final_df = merged.rename(columns={
        'full_x': 'chosen',   # 모범 답안
        'full_y': 'rejected'  # 개악된 답안
    })

    final_df = final_df[['id', 'prompt', 'chosen', 'rejected', 'gt']]

    category = pd.read_csv(category_path)
    final_df = pd.merge(final_df, category[['id', 'category']], on='id', how='left')

    from sklearn.model_selection import train_test_split

    train, valid = train_test_split(
        final_df,
        test_size=0.1,
        random_state=42,
        stratify=final_df['category']
    )

    train.to_csv(train_path, index=False, encoding='utf-8-sig')
    valid.to_csv(valid_path, index=False, encoding='utf-8-sig')
    
def main():
    merged = merge_c_r('/data/ephemeral/nlp_workspace/pro-nlp-generationfornlp-nlp-05/data/ax_y1.csv',
                       '/data/ephemeral/nlp_workspace/pro-nlp-generationfornlp-nlp-05/data/ax_y2.csv')
    add_category_split(merged, '/data/ephemeral/nlp_workspace/data/train_with_category.csv',
                       '/data/ephemeral/nlp_workspace/pro-nlp-generationfornlp-nlp-05/data/rmboost_train.csv',
                       '/data/ephemeral/nlp_workspace/pro-nlp-generationfornlp-nlp-05/data/rmboost_valid.csv')
