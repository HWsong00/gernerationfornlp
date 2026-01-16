import argparse
import pandas as pd
import ast
import json
import re
import torch
from tqdm import tqdm
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from sklearn.metrics import f1_score

# --- System Prompt ---
SYSTEM_PROMPT = """당신은 대한민국 대학수학능력시험(CSAT), KMMLU, KLUE 등 고난도 학술 및 공인 시험을 해결하는 전문 AI입니다. 
다음 지침에 따라 문제를 분석하고 해결하십시오.

1. **심층 분석 (Thinking Process)**:
- <think> 태그 안에서 지문의 핵심 논리, 역사적 맥락, 사회적 관계를 단계별로 분석하십시오.
- 각 선택지(1~5번)가 왜 정답이거나 오답인지 구체적인 근거를 지문에서 찾아 제시하십시오 (소거법 사용).

2. **최종 출력 (Final Output)**:
- 태그 외부에는 오직 "정답: [숫자]" 형식만 출력하십시오.
"""

def parse_dataset(file_path):
    print(f"📂 Loading: {file_path}")
    df = pd.read_csv(file_path)
    parsed_rows = []
    
    for idx, row in df.iterrows():
        try:
            context = row['paragraph'] if pd.notna(row['paragraph']) else ""
            problem_data = ast.literal_eval(row['problems']) if isinstance(row['problems'], str) else row['problems']
            
            main_q = problem_data['question']
            if pd.notna(row.get('question_plus')) and str(row.get('question_plus')).strip():
                main_q += f"\n\n<보기>\n{row['question_plus']}"
                
            formatted_choices = "\n".join([f"{i+1}. {c}" for i, c in enumerate(problem_data['choices'])])
            full_q = f"{main_q}\n\n[선택지]\n{formatted_choices}"
            
            parsed_rows.append({
                "original_index": idx,
                "context": context,
                "question_display": full_q,
                "ground_truth": int(problem_data['answer'])
            })
        except Exception:
            continue
            
    return pd.DataFrame(parsed_rows)

def extract_answer(text):
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    match = re.search(r'(?:정답|답|Answer|Choice)\s*[:：]?\s*([1-5])', text, re.IGNORECASE)
    if match: return int(match.group(1))
    nums = re.findall(r'([1-5])', text)
    return int(nums[-1]) if nums else None

def run_inference(adapter_path, input_csv, output_jsonl):
    # Load Model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=16384,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")

    # Load Data
    val_df = parse_dataset(input_csv)
    
    results = []
    
    print("🚀 Starting Inference...")
    for i, row in tqdm(val_df.iterrows(), total=len(val_df)):
        full_input = f"지문:\n{row['context']}\n\n문제:\n{row['question_display']}"
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": full_input}]
        
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        
        try:
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=8192,
                temperature=0.6,
                use_cache=True
            )
            raw_output = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=False).strip()
            raw_output = raw_output.replace("<|im_start|>", "").replace("<|im_end|>", "")
            pred = extract_answer(raw_output)
        except Exception as e:
            raw_output = str(e)
            pred = None

        record = {
            "index": row['original_index'],
            "ground_truth": row['ground_truth'],
            "prediction": pred,
            "is_correct": pred == row['ground_truth'],
            "raw_output": raw_output
        }
        results.append(record)

        # Periodic Save (Professional Practice)
        if i % 10 == 0:
            with open(output_jsonl, "w", encoding="utf-8") as f:
                for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Final Stats
    y_true = [r['ground_truth'] for r in results]
    y_pred = [r['prediction'] if r['prediction'] is not None else -1 for r in results]
    acc = sum(r['is_correct'] for r in results) / len(results) * 100
    f1 = f1_score(y_true, y_pred, labels=[1,2,3,4,5], average='macro')
    
    print(f"\n📊 Accuracy: {acc:.2f}% | Macro F1: {f1:.4f}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, required=True, help="Path to trained adapter")
    parser.add_argument("--input", type=str, default="train.csv")
    parser.add_argument("--output", type=str, default="predictions.jsonl")
    
    args = parser.parse_args()
    run_inference(args.adapter, args.input, args.output)