#!/usr/bin/env python3
"""
LoRA fine-tuning on the full diabetes-safe Food.com dataset,
training the model to generate both ingredients and short instructions.

- Input:   taken from the 'prompt' column
- Target:  taken from the 'target' column

This uses all rows in data/processed/t5_train_safe_full_v2.csv.
"""

from pathlib import Path
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

MODEL_NAME = "google/flan-t5-base"
BASE_DIR = Path(__file__).resolve().parents[1]

# 👇 updated to your new T5 training file
SAFE_PATH = BASE_DIR / "data" / "processed" / "t5_train_safe_full_v2.csv"

OUT_DIR = BASE_DIR / "models" / "lora_t5_base_safe_full_v2"

print(f"Loading T5-style safe recipes from: {SAFE_PATH}")
df = pd.read_csv(SAFE_PATH)

print(f"Total training rows: {len(df)}")

# Expect columns: 'prompt', 'target'
data = {
    "input_text": df["prompt"].astype(str).tolist(),
    "target_text": df["target"].astype(str).tolist(),
}

dataset = Dataset.from_dict(data)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# LoRA config (same style as before)
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"\nTrainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

def tokenize_fn(batch):
    model_inputs = tokenizer(
        batch["input_text"],
        max_length=64,
        truncation=True,
        padding=False,
    )
    labels = tokenizer(
        text_target=batch["target_text"],
        max_length=256,
        truncation=True,
        padding=False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenizing...")
tokenized = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=["input_text", "target_text"],
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = TrainingArguments(
    output_dir=str(OUT_DIR),
    per_device_train_batch_size=4,
    num_train_epochs=2,          # bump if you want
    learning_rate=5e-4,
    warmup_steps=500,
    logging_steps=200,
    save_strategy="epoch",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
    processing_class=tokenizer,
)

print("\n" + "=" * 60)
print("FULL SAFE DATASET LORA FINE-TUNING (USING T5 TRAIN FILE V2)")
print("=" * 60)
print(f"Training samples: {len(tokenized)}")
print(f"Epochs: {training_args.num_train_epochs}")
print(f"Trainable params: {100*trainable/total:.2f}%")
print("=" * 60 + "\n")

trainer.train()

print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)
OUT_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print(f"✅ Model saved to {OUT_DIR}!")

