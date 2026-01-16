# src/data.py
import os
import random
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

def prepare_dataset(cfg):
    """
    Merges local user data with external reasoning data (OLAIR/LogicKor),
    formats them using the model's chat template, and saves to disk.
    """
    output_file = cfg['data']['train_file']
    local_file = cfg['data']['input_local']
    model_id = cfg['model']['name']
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print("🚀 Initializing Tokenizer for formatting...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 1. Load User Gold Data (Local)
    print(f"📂 Loading local dataset ({local_file})...")
    my_texts = []
    if os.path.exists(local_file):
        try:
            my_dataset = load_dataset("json", data_files=local_file, split="train")
            my_texts = [
                tokenizer.apply_chat_template(row["messages"], tokenize=False)
                for row in my_dataset
            ]
            print(f"   ✅ User Data loaded: {len(my_texts)} rows")
        except Exception as e:
            print(f"   ⚠️ Error loading local file: {e}")
    else:
        print(f"   ⚠️ Local file {local_file} not found. Skipping.")

    # 2. Load Reasoning Data (Remote)
    print("☁️ Loading remote reasoning dataset...")
    r1_texts = []
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
        print("   ✅ OLAIR dataset loaded.")
    except Exception:
        print("   ⚠️ OLAIR not found, falling back to LogicKor...")
        try:
            r1_dataset = load_dataset("maywell/LogicKor", split="train")
            r1_texts = [
                tokenizer.apply_chat_template([
                    {"role": "user", "content": row['questions'][0]},
                    {"role": "assistant", "content": row['references'][0]}
                ], tokenize=False)
                for row in r1_dataset
            ]
            print("   ✅ LogicKor dataset loaded.")
        except Exception as e:
            print(f"   ❌ Failed to load external data: {e}")

    # Sampling
    sample_size = cfg['data']['sample_size_external']
    if len(r1_texts) > sample_size:
        r1_texts = random.sample(r1_texts, sample_size)
    
    # 3. Merge & Save
    all_texts = my_texts + r1_texts
    random.shuffle(all_texts)

    print(f"💾 Saving {len(all_texts)} rows to {output_file}...")
    df = pd.DataFrame({"text": all_texts})
    df.to_json(output_file, orient="records", lines=True, force_ascii=False)
    
    return output_file