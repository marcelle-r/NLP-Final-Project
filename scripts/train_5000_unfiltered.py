from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model
import torch
import pandas as pd
import ast

MODEL_NAME = "google/flan-t5-base"

# give this run a distinct name so it doesn't clash with the safe-LoRA run
RUN_NAME = "lora_t5_base_5000_unfiltered"
MODEL_DIR = f"models/{RUN_NAME}"

# === UNFILTERED TRAINING: sample 5000 recipes from RAW_recipes ===
df = pd.read_csv("data/RAW_recipes.csv").sample(5000, random_state=42)

data = {"input_text": [], "target_text": []}

print("Formatting 5000 UNFILTERED recipes...")
for _, row in df.iterrows():
    try:
        ingredients = ast.literal_eval(row["ingredients"])
        title = row["name"]

        # same diabetes-friendly prompt; only the training data is unfiltered
        inp = f"Generate a diabetes-friendly {title} recipe."
        tgt = f"Ingredients: {', '.join(ingredients[:8])}"

        data["input_text"].append(inp)
        data["target_text"].append(tgt)
    except Exception:
        continue

print(f"Created {len(data['input_text'])} training examples")

dataset = Dataset.from_dict(data)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# LoRA config (same as safe run)
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
print(f"\nTrainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

def tokenize_fn(batch):
    model_inputs = tokenizer(
        batch["input_text"], max_length=64, truncation=True, padding=False
    )
    labels = tokenizer(
        text_target=batch["target_text"],
        max_length=128,
        truncation=True,
        padding=False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenizing...")
tokenized = dataset.map(
    tokenize_fn, batched=True, remove_columns=["input_text", "target_text"]
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    per_device_train_batch_size=4,
    num_train_epochs=3,  # 3 epochs for 5K recipes
    logging_steps=100,
    save_strategy="no",
    report_to="none",
    learning_rate=5e-4,
    warmup_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
    processing_class=tokenizer,
)

print("\n" + "=" * 60)
print("UNFILTERED LORA FINE-TUNING - FULL SCALE")
print("=" * 60)
print(f"Model: T5-Base with LoRA (r=8)")
print(f"Training samples: {len(tokenized)}")
print("Dataset: RAW_recipes (no diabetes filter)")
print(f"Epochs: 3")
print(f"Trainable params: {100 * trainable / total:.2f}%")
print("=" * 60 + "\n")

trainer.train()

print("\n" + "=" * 60)
print("SAVING UNFILTERED MODEL")
print("=" * 60)
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print(f"✅ Unfiltered model saved to {MODEL_DIR}!")

