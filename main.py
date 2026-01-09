# main.py
import yaml
import argparse
import sys
import os

# Ensure the current directory is in the path so we can import 'src'
sys.path.append(os.getcwd())

from src.data import prepare_dataset
from src.model import load_model
from src.trainer import run_training

def load_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Professional Qwen3 Finetuning Pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML config")
    parser.add_argument("--skip_data", action="store_true", help="Skip data preparation if file exists")
    args = parser.parse_args()

    # 1. Load Configuration
    print(f"⚙️  Loading configuration from {args.config}...")
    cfg = load_config(args.config)

    # 2. Data Preparation
    if not args.skip_data:
        print("\n--- [Step 1/3] Data Preparation ---")
        prepare_dataset(cfg)
    else:
        print("\n--- [Step 1/3] Skipping Data Preparation (User Request) ---")

    # 3. Model Loading
    print("\n--- [Step 2/3] Model Loading ---")
    model, tokenizer = load_model(cfg)

    # 4. Training
    print("\n--- [Step 3/3] Training Execution ---")
    run_training(model, tokenizer, cfg)

if __name__ == "__main__":
    main()