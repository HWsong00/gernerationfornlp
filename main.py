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

# Updated load_config for main.py
def load_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at: {path}")
    
    # 1. Load Main Config (Training & Data)
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # 2. Load Modular Model Config
    # If the main config points to a model config, load and merge it
    if 'global' in cfg and 'model_config_path' in cfg['global']:
        model_path = cfg['global']['model_config_path']
        if os.path.exists(model_path):
            print(f"🔗 Merging model config from: {model_path}")
            with open(model_path, 'r', encoding='utf-8') as f:
                model_cfg = yaml.safe_load(f)
                # Merge the model dictionary into the main config
                cfg.update(model_cfg)
        else:
            print(f"⚠️ Warning: Model config path '{model_path}' not found.")
            
    return cfg

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