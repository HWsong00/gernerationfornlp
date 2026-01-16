# src/model.py
import torch
from unsloth import FastLanguageModel

def load_model(cfg):
    """
    Loads the Quantized Qwen Model and attaches LoRA adapters based on config.
    """
    print(f"🤖 Loading Model: {cfg['model']['name']}...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = cfg['model']['name'],
        max_seq_length = cfg['model']['max_seq_length'],
        dtype = None,
        load_in_4bit = cfg['model']['load_in_4bit'],
    )

    print("🔧 Attaching LoRA Adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = cfg['lora']['r'],
        target_modules = cfg['lora']['target_modules'],
        lora_alpha = cfg['lora']['alpha'],
        lora_dropout = cfg['lora']['dropout'],
        bias = cfg['lora']['bias'],
        use_gradient_checkpointing = "unsloth",
        random_state = cfg['global']['seed'],
    )
    
    return model, tokenizer