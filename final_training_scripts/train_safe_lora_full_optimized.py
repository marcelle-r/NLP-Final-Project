#!/usr/bin/env python3
"""
OPTIMIZED Safe LoRA - Full (62K recipes)
- Batch size: 12 (effective: 48)
- Learning rate: 3e-4
- fp16 disabled (prevents NaN)
- Auto-push every 1000 steps
Expected: ~2 hours on T4
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

BASE_DIR = Path("/content/NLP-Final-Project")
TRAINING_DATA_DIR = BASE_DIR / "final_training_data"
OUTPUT_DIR = BASE_DIR / "final_models" / "safe_lora_full"

BASE_MODEL = "google/flan-t5-base"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 512

# LoRA
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

# Training (OPTIMIZED)
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 3e-4
NUM_EPOCHS = 3
WARMUP_STEPS = 500
LOGGING_STEPS = 50
SAVE_STEPS = 1000
PUSH_STEPS = 1000

print("="*70)
print("TRAINING: SAFE LORA FULL - OPTIMIZED")
print("="*70)

if not torch.cuda.is_available():
    print("⚠️ NO GPU!")
    exit(1)

print(f"\n✓ GPU: {torch.cuda.get_device_name(0)}")

# Auto-push callback
class GitHubPushCallback(TrainerCallback):
    def __init__(self, push_every=1000):
        self.push_every = push_every
        self.last_push = 0
    
    def on_save(self, args, state, control, **kwargs):
        step = state.global_step
        if step - self.last_push >= self.push_every:
            print(f"\n📤 Step {step}: Pushing...")
            try:
                subprocess.run(["git", "add", str(OUTPUT_DIR)], cwd=BASE_DIR, check=True, capture_output=True)
                subprocess.run(["git", "commit", "-m", f"Step {step}"], cwd=BASE_DIR, check=False, capture_output=True)
                subprocess.run(["git", "push"], cwd=BASE_DIR, capture_output=True, timeout=30)
                print("✓ Pushed")
                self.last_push = step
            except:
                print("⚠️ Push failed")

# Load data
print("\n1. Loading data...")
def load_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [l.strip() for l in f if l.strip()]

train_prompts = load_file(TRAINING_DATA_DIR / "safe_full_train_prompts.txt")
train_targets = load_file(TRAINING_DATA_DIR / "safe_full_train_targets.txt")
val_prompts = load_file(TRAINING_DATA_DIR / "safe_full_val_prompts.txt")
val_targets = load_file(TRAINING_DATA_DIR / "safe_full_val_targets.txt")

print(f"✓ Train: {len(train_prompts):,}")
print(f"✓ Val: {len(val_prompts):,}")

train_dataset = Dataset.from_dict({'input': train_prompts, 'target': train_targets})
val_dataset = Dataset.from_dict({'input': val_prompts, 'target': val_targets})

# Load model
print("\n2. Loading model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
print(f"✓ {BASE_MODEL}")

# LoRA
print("\n3. Applying LoRA...")
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q", "v"]
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Tokenize
print("\n4. Tokenizing...")
def preprocess(examples):
    inputs = tokenizer(examples['input'], max_length=MAX_INPUT_LENGTH, truncation=True, padding=False)
    labels = tokenizer(examples['target'], max_length=MAX_TARGET_LENGTH, truncation=True, padding=False)
    inputs["labels"] = labels["input_ids"]
    return inputs

train_dataset = train_dataset.map(preprocess, batched=True, remove_columns=['input', 'target'])
val_dataset = val_dataset.map(preprocess, batched=True, remove_columns=['input', 'target'])
print("✓ Done")

# Setup trainer
print("\n5. Configuring...")
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
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    predict_with_generate=False,
    fp16=False,
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
    callbacks=[GitHubPushCallback(PUSH_STEPS)]
)

print("✓ Ready")

# Train
print("\n6. Training...")
print(f"   Batch: {BATCH_SIZE} × {GRADIENT_ACCUMULATION_STEPS} = {BATCH_SIZE*GRADIENT_ACCUMULATION_STEPS}")
print(f"   LR: {LEARNING_RATE}")
print(f"   Epochs: {NUM_EPOCHS}")
print()

start = time.time()
trainer.train()
elapsed = time.time() - start

print(f"\n✓ Done: {elapsed/60:.1f} min")

# Save
print("\n7. Saving...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

with open(OUTPUT_DIR / "log.txt", 'w') as f:
    f.write(f"Time: {elapsed/60:.1f}min\n")
    f.write(f"Steps: {trainer.state.global_step}\n")

# Final push
print("\n8. Final push...")
try:
    subprocess.run(["git", "add", str(OUTPUT_DIR)], cwd=BASE_DIR, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", f"Final ({elapsed/60:.1f}min)"], cwd=BASE_DIR, check=False, capture_output=True)
    subprocess.run(["git", "push"], cwd=BASE_DIR, capture_output=True, timeout=30)
    print("✓ Pushed")
except:
    print("⚠️ Failed")

print("\n" + "="*70)
print("✅ COMPLETE")
print("="*70)
