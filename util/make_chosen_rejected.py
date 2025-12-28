# Request
from openai import OpenAI
import pandas as pd
import ast
from tqdm import tqdm
import random
import re

client = OpenAI(
    base_url="https://guest-api.sktax.chat/v1",
    api_key="sktax-XyeKFrq67ZjS4EpsDlrHHXV8it"  # 공개 게스트 키
)

Y1_PROMPT = """
너는 대한민국 최고의 명문대학교에서 수능형 문항 분석과 교수법을 담당하는 교육학 교수이다.
너의 목표는 학생들에게 복잡한 수능형 지문과 문제를 어떻게 논리적으로 풀고 분석해야 하는지 가르치는 것이다.
단순히 정답만 맞히는 것이 아니라, 지문에 근거하여 오답이 왜 오답인지까지 명확하게 설명하는 '가장 완벽한 해설'을 제공해야 한다.

<task_description>
아래에 수능형 문제의 평가 가이드라인과 실제 문항(지문 및 발문)이 제시될 것이다. 
너는 이 가이드라인을 엄격히 준수하여 해당 문제에 대한 정답과 심층적인 풀이 과정을 작성해야 한다.
</task_description>

<guideline>
수능형 문제의 풀이는 다음의 세 가지 관점에서 평가된다:
- (Logical Grounding, 논리적 근거): 모든 풀이는 반드시 제시된 지문(또는 자료)의 구체적인 문장이나 정보에 근거해야 한다. 주관적인 배경지식에 의존하기보다 지문 내에서 답의 근거를 찾아 연결해야 한다.
- (Conceptual Accuracy, 개념적 정확성): 국어, 사회, 역사 등 각 과목의 핵심 개념과 용어를 정확하게 사용해야 한다. 특히 역사적 사건의 인과관계나 사회적 현상의 정의가 틀림이 없어야 한다.
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
학생들에게 모범이 될 수 있도록 논리적이고 권위 있는 해설을 작성하라.
작성된 해설은 반드시 <response></response> 태그 안에 넣어라.
해설이 끝난 후, 반드시 맨 마지막 줄에 최종 정답을 {{"정답": n}} 형식의 문자열을 추가하라.
""".strip()

Y2_SYS_PROMPT = """
너는 대한민국 최고의 명문대학교에서 수능형 문항 분석과 교수법을 담당하는 교육학 교수이다. 
너의 목표는 학생들에게 수능 지문과 문항을 분석하는 법을 가르치는 것이다.
너는 질문, 지문, 그리고 이미 작성된 '모범 해설'을 검토한 뒤, 특정 측면에서 이보다 품질이 낮은 '비교용 해설'을 만들어 학생들의 비판적 사고를 돕고자 한다.
""".strip()

Y2_USER_PROMPT = """
<task_description>
아래에 해설의 세부 평가 항목이 포함된 가이드라인이 제시될 것이다. 
그다음 질문, 지문 그리고 이미 작성된 모범 해설이 제공된다. 
너는 다음 단계를 수행해야 한다.

Step 1: 가이드라인에서 품질을 낮출 항목을 몇 가지 선택하라.
Step 2: 선택한 항목 측면에서 모범 해설보다 품질이 낮은 새로운 해설을 생성하라.

[지시 사항]
- 제작되는 비교용 해설은 모범 해설의 문체, 어조, 그리고 문단 구조를 완벽하게 복제해야 한다.
- 오직 '내용의 논리적 결함'이나 '개념적 오류'만이 유일한 차이점이 되어야 하며, 겉모습만으로는 모범 해설과 구별할 수 없을 만큼 매력적인 오답이어야 한다.
- 이 해설이 의도적으로 품질을 낮춘 것이라는 사실을 절대 언급하지 마라.
- - 중요!: 선지의 {n}번을 논리가 올바른 정답인 것처럼 서술하라.
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

[모범 답안]
{good_answer}
</context_and_question>

task_description과 guideline을 숙지하라.
작성된 '비교용 해설'은 반드시 <response></response> 태그 안에 넣어라.
해설이 끝난 후 맨 마지막 줄에 '비교용 해설' 기준의 최종 정답을 {{"정답": n}} 형식으로 추가하라.
선지의 {n}번을 논리가 올바른 정답인 것처럼 서술하라.
""".strip()


def parse_dataset(file_path):
    print(f"{file_path} 로딩")
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

            q_plus = row.get('question_plus')
            if pd.notna(q_plus) and str(q_plus).strip():
                main_question += f"\n\n<보기>\n{q_plus}"

            formatted_choices = "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(choices)])
            full_question_display = f"{main_question}\n\n[선택지]\n{formatted_choices}"

            parsed_rows.append({
                "id": row_id,
                "context": context,
                "question_display": full_question_display,
                "answer": problem_data['answer'],
                "choices_len": len(choices)
            })
        except Exception as e:
            print(f"Warning: Error parsing row {idx}: {e}")
            continue

    return pd.DataFrame(parsed_rows)

def extract_answer(text):
  text = text.strip()
  try:
        # 텍스트 뒤에서부터 '{'를 찾아서 JSON/Dict 파싱 시도
        start_idx = text.rfind('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            dict_str = text[start_idx : end_idx+1]
            data = ast.literal_eval(dict_str) # 문자열을 딕셔너리로 변환
            if '정답' in data:
                return int(data['정답'])
        else: return 10
  except Exception:
        return 10
    
def make_chosen(data, output_path):
    msg = Y1_PROMPT
    results = []
    
    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Generating"):
        row_id = row['id']
        context = row['context']
        question = row['question_display']
        answer = row['answer']

        completion = client.chat.completions.create(
            model="ax4",
            messages=[
                {"role": "user", "content": msg.format(context=context, question=question)}
            ]
        )

        response = completion.choices[0].message.content
        pred = extract_answer(response)
        
        # 결과 저장 구조
        results.append({
            "id": row_id,
            "input_context": context,
            "input_question": question,
            "full": response,
            "model_answer": pred,
            "gt": answer,
            "is_correct": str(pred) == str(answer)
        })

        # [안전장치] 10문제마다 파일 덮어쓰기 (중간 저장)
        if len(results) % 10 == 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        output = pd.DataFrame(results)
        output.to_csv(output_path, index=False, encoding='utf-8-sig')


def make_rejected(chosen_path, data, output_path):
    chosen = pd.read_csv(chosen_path)
    
    results = []
    for idx, row in tqdm(chosen.iterrows(), total=len(chosen), desc="Generating"):
        choices_len = data.loc[data['id'] == row['id'], 'choices_len'].values[0]
        options = [i for i in range(1, choices_len+1)]
        options.remove(row['gt'])
        
        random_number = random.choice(options)
        
        row_id = row['id']
        context = row['input_context']
        question = row['input_question']
        good_answer = row['full']
        gt = row['gt']

        completion = client.chat.completions.create(
            model="ax4",
            messages=[
                {"role": "system", "content": Y2_SYS_PROMPT},
                {"role": "user", "content": Y2_USER_PROMPT.format(context=context, 
                                                                  question=question, 
                                                                  good_answer=good_answer,
                                                                  n=random_number)}
            ]
        )

        response = completion.choices[0].message.content
        pred = extract_answer(response)
        
        # 결과 저장 구조
        results.append({
            "id": row_id,
            "input_context": context,
            "input_question": question,
            "full": response,
            "model_answer": pred,
            "gt": gt,
            "is_correct": str(pred) == str(gt)
        })

        # [안전장치] 10문제마다 파일 덮어쓰기 (중간 저장)
        if len(results) % 10 == 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        output = pd.DataFrame(results)
        output.to_csv(output_path, index=False, encoding='utf-8-sig')


def main():
    data = parse_dataset('/data/ephemeral/nlp_workspace/data/train.csv')
    condition = input("chosen 혹은 rejected: ")
    
    if condition == "chosen":
        make_chosen(data, 'ax.csv')
        
    elif condition == "rejected":
        chosen_path = '/data/ephemeral/nlp_workspace/pro-nlp-generationfornlp-nlp-05/eda/ax_correct.csv'
        make_rejected(chosen_path, data, '/data/ephemeral/nlp_workspace/ax_y2.csv' )
        
    else:
        print("해당하는 함수가 없습니다.")
        
    
if __name__ == "__main__":
    main()