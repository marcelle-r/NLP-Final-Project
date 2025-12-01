#!/usr/bin/env python3
"""
LoRA fine-tuning on the full diabetes-safe Food.com dataset,
training the model to generate both ingredients and short instructions.

- Input:   "Generate a diabetes-friendly recipe called {name}."
- Target:  "Ingredients: ... Instructions: ..."

This uses all rows in data/processed/foodcom_safe_recipes.csv.
"""

from pathlib import Path
import ast
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
SAFE_PATH = BASE_DIR / "data" / "processed" / "foodcom_safe_recipes.csv"
OUT_DIR = BASE_DIR / "models" / "lora_t5_base_safe_full"

print(f"Loading safe recipes from: {SAFE_PATH}")
df = pd.read_csv(SAFE_PATH)

print(f"Total safe recipes: {len(df)}")

data = {"input_text": [], "target_text": []}

for _, row in df.iterrows():
    name = str(row.get("name", "")).strip()
    if not name:
        continue

    ingredients = str(row.get("ingredients", "")).strip()
    steps_raw = row.get("steps", "")

    # Build prompt
    inp = f"Generate a diabetes-friendly recipe called {name}."

    # Try to interpret steps as a list, otherwise treat as plain text
    step_text = ""
    if isinstance(steps_raw, str) and steps_raw:
        try:
            maybe_list = ast.literal_eval(steps_raw)
            if isinstance(maybe_list, (list, tuple)):
                step_text = " ".join(str(s) for s in maybe_list[:3])
            else:
                step_text = steps_raw
        except Exception:
            step_text = steps_raw

    tgt = f"Ingredients: {ingredients}"
    if step_text:
        tgt += f" Instructions: {step_text}"

    data["input_text"].append(inp)
    data["target_text"].append(tgt)

print(f"Created {len(data['input_text'])} training examples")

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
    num_train_epochs=2,          # you can bump to 3 if your Mac survives :)
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
print("FULL SAFE DATASET LORA FINE-TUNING (INGREDIENTS + INSTRUCTIONS)")
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

