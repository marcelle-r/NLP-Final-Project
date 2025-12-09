#!/usr/bin/env python3
"""
Train Unfiltered LoRA 5K - no safety filtering applied to training data.
"""
import torch
import gc
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import pandas as pd
import ast

BASE_DIR = Path(__file__).resolve().parents[1]

def format_ingredients(ing_list):
    if isinstance(ing_list, str):
        try:
            ing_list = ast.literal_eval(ing_list)
        except:
            return str(ing_list)
    if isinstance(ing_list, list):
        return ', '.join(str(i) for i in ing_list)
    return str(ing_list)

def format_steps(steps_list):
    if isinstance(steps_list, str):
        try:
            steps_list = ast.literal_eval(steps_list)
        except:
            return str(steps_list)
    if isinstance(steps_list, list):
        return ' '.join([f"{i+1}. {step}" for i, step in enumerate(steps_list)])
    return str(steps_list)

def main():
    print("Training Unfiltered LoRA 5K")
    
    # Clear GPU
    torch.cuda.empty_cache()
    gc.collect()
    
    # Load UNFILTERED data
    raw_data = pd.read_csv(BASE_DIR / "data/raw/RAW_recipes.csv")
    sampled = raw_data.sample(n=5000, random_state=42)
    
    train_size = int(0.9 * len(sampled))
    train_df = sampled.iloc[:train_size].reset_index(drop=True)
    val_df = sampled.iloc[train_size:].reset_index(drop=True)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    # Format data
    train_prompts = []
    train_targets = []
    for _, row in train_df.iterrows():
        prompt = f"Generate a diabetes-friendly {row['name']} recipe."
        ingredients = format_ingredients(row['ingredients'])
        steps = format_steps(row['steps'])
        target = f"Ingredients: {ingredients} Instructions: {steps}"
        train_prompts.append(prompt)
        train_targets.append(target)
    
    val_prompts = []
    val_targets = []
    for _, row in val_df.iterrows():
        prompt = f"Generate a diabetes-friendly {row['name']} recipe."
        ingredients = format_ingredients(row['ingredients'])
        steps = format_steps(row['steps'])
        target = f"Ingredients: {ingredients} Instructions: {steps}"
        val_prompts.append(prompt)
        val_targets.append(target)
    
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
        output_dir=str(BASE_DIR / "final_models/unfiltered_lora_5k"),
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
    model.save_pretrained(BASE_DIR / "final_models/unfiltered_lora_5k")
    tokenizer.save_pretrained(BASE_DIR / "final_models/unfiltered_lora_5k")
    
    print("✅ Unfiltered LoRA 5K complete!")

if __name__ == "__main__":
    main()
