# src/trainer.py
import torch
import gc
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

def run_training(model, tokenizer, cfg):
    """
    Initializes SFTTrainer with the loaded model and config, then runs training.
    """
    dataset_path = cfg['data']['train_file']
    output_dir = cfg['global']['output_dir']
    
    print(f"📚 Loading prepared dataset from {dataset_path}...")
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    print(f"🚀 Initializing SFT Trainer (Epochs: {cfg['training']['epochs']})...")
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = cfg['model']['max_seq_length'],
        packing = False,
        args = SFTConfig(
            per_device_train_batch_size = cfg['training']['batch_size'],
            gradient_accumulation_steps = cfg['training']['gradient_accumulation'],
            num_train_epochs = cfg['training']['epochs'],
            learning_rate = cfg['training']['lr'],
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = cfg['training']['optim'],
            weight_decay = cfg['training']['weight_decay'],
            lr_scheduler_type = cfg['training']['lr_scheduler'],
            seed = cfg['global']['seed'],
            output_dir = output_dir,
            gradient_checkpointing = True,
            save_strategy = "epoch",
            save_total_limit = 3,
            neftune_noise_alpha = 5,
        ),
    )

    # Train
    print("🔥 Starting Training...")
    trainer.train()
    
    # Save Final Model
    print(f"💾 Saving final adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Cleanup
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    print("✅ Training Complete!")