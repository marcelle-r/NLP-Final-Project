#!/usr/bin/env python3
"""
Train Safe LoRA - Full (62K filtered recipes)
Auto-saves and pushes checkpoints to GitHub every 500 steps
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from pathlib import Path
import time
import subprocess

# ==================== CONFIG ====================

BASE_DIR = Path("/content/NLP-Final-Project")
TRAINING_DATA_DIR = BASE_DIR / "final_training_data"
OUTPUT_DIR = BASE_DIR / "final_models" / "safe_lora_full"

BASE_MODEL = "google/flan-t5-base"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 512

# LoRA config
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

# Training config
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3
WARMUP_STEPS = 500
LOGGING_STEPS = 100
SAVE_STEPS = 500
PUSH_CHECKPOINT_STEPS = 500

print("="*70)
print("TRAINING: SAFE LORA - FULL (62K recipes)")
print("="*70)

if torch.cuda.is_available():
    print(f"\n✓ GPU: {torch.cuda.get_device_name(0)}")
else:
    print("\n⚠️ WARNING: No GPU detected!")

# ==================== GITHUB AUTO-PUSH CALLBACK ====================

class GitHubPushCallback(TrainerCallback):
    """Automatically push checkpoints to GitHub."""
    
    def __init__(self, push_every_steps=500):
        self.push_every_steps = push_every_steps
        self.last_push_step = 0
    
    def on_save(self, args, state, control, **kwargs):
        current_step = state.global_step
        
        if current_step - self.last_push_step >= self.push_every_steps:
            print(f"\n📤 Pushing checkpoint at step {current_step}...")
            try:
                subprocess.run(["git", "add", str(OUTPUT_DIR)], cwd=BASE_DIR, check=True, capture_output=True)
                subprocess.run(["git", "commit", "-m", f"Checkpoint: step {current_step}"], cwd=BASE_DIR, check=False, capture_output=True)
                result = subprocess.run(["git", "push"], cwd=BASE_DIR, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"✓ Pushed!")
                else:
                    print(f"⚠️ Push failed")
                    
                self.last_push_step = current_step
            except Exception as e:
                print(f"⚠️ Error: {e}")

# ==================== LOAD DATA ====================

print("\n1. Loading training data...")

def load_text_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

train_prompts = load_text_file(TRAINING_DATA_DIR / "safe_full_train_prompts.txt")
train_targets = load_text_file(TRAINING_DATA_DIR / "safe_full_train_targets.txt")
val_prompts = load_text_file(TRAINING_DATA_DIR / "safe_full_val_prompts.txt")
val_targets = load_text_file(TRAINING_DATA_DIR / "safe_full_val_targets.txt")

print(f"✓ Training examples: {len(train_prompts):,}")
print(f"✓ Validation examples: {len(val_prompts):,}")

train_dataset = Dataset.from_dict({'input': train_prompts, 'target': train_targets})
val_dataset = Dataset.from_dict({'input': val_prompts, 'target': val_targets})

# ==================== LOAD MODEL ====================

print("\n2. Loading base model and tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

print(f"✓ Loaded {BASE_MODEL}")

# ==================== SETUP LORA ====================

print("\n3. Configuring LoRA...")

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q", "v"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ==================== PREPROCESSING ====================

print("\n4. Preprocessing data...")

def preprocess_function(examples):
    model_inputs = tokenizer(examples['input'], max_length=MAX_INPUT_LENGTH, truncation=True, padding=False)
    labels = tokenizer(examples['target'], max_length=MAX_TARGET_LENGTH, truncation=True, padding=False)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=['input', 'target'])
val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=['input', 'target'])

print("✓ Data preprocessed")

# ==================== TRAINING SETUP ====================

print("\n5. Setting up training...")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

training_args = Seq2SeqTrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    eval_steps=SAVE_STEPS,
    eval_strategy="steps",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    predict_with_generate=False,
    fp16=False,  # Disabled - may cause issues with LoRA
    label_smoothing_factor=0.1,
    report_to="none",
    push_to_hub=False
)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    callbacks=[GitHubPushCallback(push_every_steps=PUSH_CHECKPOINT_STEPS)]
)

print("✓ Trainer configured with auto-push")

# ==================== TRAIN ====================

print("\n6. Starting training...")
print(f"   Checkpoints: Every {SAVE_STEPS} steps")
print(f"   GitHub push: Every {PUSH_CHECKPOINT_STEPS} steps")
print()

start_time = time.time()
trainer.train()
elapsed = time.time() - start_time

print(f"\n✓ Training completed in {elapsed/60:.1f} minutes")

# ==================== SAVE FINAL ====================

print("\n7. Saving final model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Final push
print("\n8. Final push to GitHub...")
try:
    subprocess.run(["git", "add", str(OUTPUT_DIR)], cwd=BASE_DIR, check=True)
    subprocess.run(["git", "commit", "-m", f"Final: safe_lora_full ({elapsed/60:.1f}min)"], cwd=BASE_DIR, check=False)
    subprocess.run(["git", "push"], cwd=BASE_DIR)
    print("✓ Pushed!")
except:
    pass

print("\n" + "="*70)
print("✅ TRAINING COMPLETE")
print("="*70)
