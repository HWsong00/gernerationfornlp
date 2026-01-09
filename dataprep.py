import argparse
import random
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

# Constants
MODEL_ID = "unsloth/Qwen3-8B-unsloth-bnb-4bit"

def load_and_format_data(output_file, sample_size=2000):
    print("🚀 Initializing Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # 1. Load User Gold Data (Local)
    print("📂 Loading local dataset (csat_sft.jsonl)...")
    try:
        my_dataset = load_dataset("json", data_files="csat_sft.jsonl", split="train")
        my_texts = [
            tokenizer.apply_chat_template(row["messages"], tokenize=False)
            for row in my_dataset
        ]
        print(f"✅ User Data loaded: {len(my_texts)} rows")
    except Exception as e:
        print(f"⚠️ Could not load local dataset: {e}")
        my_texts = []

    # 2. Load Reasoning Data (Remote)
    print("☁️ Loading remote reasoning dataset...")
    try:
        # Priority: OLAIR -> Fallback: LogicKor
        r1_dataset = load_dataset("OLAIR/Open-R1-Ko-SFT-v2.0", split="train")
        r1_texts = [
            tokenizer.apply_chat_template([
                {"role": "user", "content": row['prompt']},
                {"role": "assistant", "content": row['response']}
            ], tokenize=False)
            for row in r1_dataset
        ]
    except Exception as e:
        print(f"⚠️ OLAIR failed ({e}), falling back to LogicKor...")
        r1_dataset = load_dataset("maywell/LogicKor", split="train")
        r1_texts = [
            tokenizer.apply_chat_template([
                {"role": "user", "content": row['questions'][0]},
                {"role": "assistant", "content": row['references'][0]}
            ], tokenize=False)
            for row in r1_dataset
        ]

    # Sampling
    if len(r1_texts) > sample_size:
        r1_texts = random.sample(r1_texts, sample_size)
    print(f"✅ Reasoning Data loaded: {len(r1_texts)} rows")

    # 3. Merge & Save
    all_texts = my_texts + r1_texts
    random.shuffle(all_texts)

    print(f"💾 Saving {len(all_texts)} rows to {output_file}...")
    df = pd.DataFrame({"text": all_texts})
    df.to_json(output_file, orient="records", lines=True, force_ascii=False)
    print("🔥 Data preparation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare datasets for Qwen3 Finetuning")
    parser.add_argument("--output", type=str, default="combined_csat_sft.jsonl", help="Output JSONL filename")
    parser.add_argument("--sample_size", type=int, default=2000, help="Number of external samples to include")
    
    args = parser.parse_args()
    
    # Note: Ensure you are logged in via `huggingface-cli login` in your terminal
    load_and_format_data(args.output, args.sample_size)