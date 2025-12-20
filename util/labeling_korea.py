import pandas as pd
from ast import literal_eval
from openai import OpenAI
import os
from dotenv import load_dotenv
from tqdm import tqdm


df = pd.read_csv('data/train_with_category.csv') 

def build_prompt(paragraph: str, question: str, choices) -> str:
    return f"""
다음은 수능 스타일의 문제이다.

[지문]
{paragraph}

[문제]
{question}

[선택지]
{choices}

### Instruction
주어진 문제가 "한국(Korea)"에 관한 특수한 배경지식, 한국의 사회·문화·역사적 맥락을 이해해야 더 정확하게 풀 수 있는 문제인지 판단하세요.

### Criteria
- 한국의 고유한 지명, 인물, 사건, 제도, 관습 등이 포함된 경우
- 한국의 교육과정이나 사회적 상황에 특화된 논리가 필요한 경우

위 조건에 해당하면 True, 보편적인 세계사적 사실이나 일반적인 논리만으로 충분히 풀 수 있다면 False를 출력하세요.

### Output Format
True 혹은 False 중 하나만 정확히 출력하라.

"""

def classify_korea(client, paragraph: str, question: str, choices) -> str:
    prompt = build_prompt(paragraph, question, choices)

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt,
        reasoning={"effort": "low"},
        text={"verbosity": "low"}
    )

    # 모델 출력 텍스트 추출
    is_korea = response.output_text.strip()

    # 안전장치: 허용 목록 외 출력 방지
    if is_korea not in ["True", "False"]:
        return "UNKNOWN"

    return is_korea

def main():
    load_dotenv('.env')  
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    tqdm.pandas()
    df["is_korea"] = df.progress_apply(
        lambda row: classify_korea(
            client,
            paragraph=row["paragraph"],
            question=row["question"],
            choices=row["choices"]
        ),
        axis=1
    )
    
    output_path = "data/train_with_korea.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
if __name__ == "__main__":
    main()
    
    