import argparse
import torch
import gc
import os
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

def train(dataset_path, output_dir, epochs, lr, max_seq_length=16384):
    print(f"\n=== 🚀 Starting Training: {output_dir} ===")
    
    # 1. Load Model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen3-8B-unsloth-bnb-4bit",
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True,
    )

    # 2. Add LoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = 64,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 128,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    # 3. Load Data
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    print(f"✅ Loaded training data: {len(dataset)} examples")

    # 4. Initialize Trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        packing = False,
        args = SFTConfig(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 16,
            num_train_epochs = epochs,
            learning_rate = lr,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = output_dir,
            gradient_checkpointing = True,
            save_strategy = "epoch",
        ),
    )

    # 5. Train & Save
    trainer.train()
    
    print(f"💾 Saving adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Cleanup
    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()
    print("✅ Training Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to the training JSONL file")
    parser.add_argument("--output", type=str, default="lora_qwen3_finetune", help="Output directory for adapters")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    
    args = parser.parse_args()
    train(args.data, args.output, args.epochs, args.lr)