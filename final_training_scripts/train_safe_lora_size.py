#!/usr/bin/env python3
"""
Train Safe LoRA models with different dataset sizes.
Usage: python train_safe_lora_size.py --size {1k,5k,10k,full}
"""
import argparse
import torch
import gc
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

BASE_DIR = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", required=True, choices=["1k", "5k", "10k", "full"])
    args = parser.parse_args()
    
    size = args.size
    print(f"Training Safe LoRA {size.upper()}")
    
    # Clear GPU
    torch.cuda.empty_cache()
    gc.collect()
    
    # Load data
    training_data_dir = BASE_DIR / "final_training_data"
    
    with open(training_data_dir / f"safe_{size}_train_prompts.txt", 'r') as f:
        train_prompts = f.read().split('\n')
    
    with open(training_data_dir / f"safe_{size}_train_targets.txt", 'r') as f:
        train_targets = f.read().split('\n')
    
    with open(training_data_dir / f"safe_{size}_val_prompts.txt", 'r') as f:
        val_prompts = f.read().split('\n')
    
    with open(training_data_dir / f"safe_{size}_val_targets.txt", 'r') as f:
        val_targets = f.read().split('\n')
    
    print(f"Loaded {len(train_prompts)} train + {len(val_prompts)} val examples")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    
    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q", "v"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Create datasets
    class SimpleDataset:
        def __init__(self, prompts, targets, tokenizer):
            self.prompts = prompts
            self.targets = targets
            self.tokenizer = tokenizer
        
        def __len__(self):
            return len(self.prompts)
        
        def __getitem__(self, idx):
            prompt = self.prompts[idx]
            target = self.targets[idx]
            
            inputs = self.tokenizer(prompt, max_length=512, truncation=True, padding=False)
            labels = self.tokenizer(target, max_length=512, truncation=True, padding=False)
            
            return {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "labels": labels["input_ids"]
            }
    
    train_dataset = SimpleDataset(train_prompts, train_targets, tokenizer)
    val_dataset = SimpleDataset(val_prompts, val_targets, tokenizer)
    
    # Data collator
    from transformers import DataCollatorForSeq2Seq
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(BASE_DIR / f"final_models/safe_lora_{size}"),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=1000,
        save_total_limit=2,
        fp16=False,
        load_best_model_at_end=False,
        report_to="none",
    )
    
    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    trainer.train()
    
    # Save
    model.save_pretrained(BASE_DIR / f"final_models/safe_lora_{size}")
    tokenizer.save_pretrained(BASE_DIR / f"final_models/safe_lora_{size}")
    
    print(f"✅ Safe LoRA {size.upper()} complete!")

if __name__ == "__main__":
    main()
