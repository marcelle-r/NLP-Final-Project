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

# New run name for THIS experiment (unfiltered + context)
RUN_NAME = "lora_t5_base_5000_unfiltered_context"
MODEL_DIR = f"models/{RUN_NAME}"

# ----- Context + in-context example for the prompts -----

CONTEXT = (
    "You are a nutritionist specializing in type 2 diabetes.\n"
    "Follow these rules when creating recipes:\n"
    "- Avoid added sugars (sugar, brown sugar, syrups, sweetened drinks, desserts).\n"
    "- Avoid refined white flours, white bread/pasta, and deep-fried foods.\n"
    "- Prefer non-nutritive sweeteners like stevia, erythritol, or monk fruit.\n"
    "- Prefer high-fiber vegetables (broccoli, spinach, zucchini, cauliflower, green beans).\n"
    "- Prefer whole grains (quinoa, brown rice, oats) and lean proteins (salmon, tuna,\n"
    "  chicken breast, turkey, tofu, Greek yogurt, cottage cheese).\n"
)

FEWSHOT_EXAMPLE = (
    "Example diabetes-friendly recipe:\n"
    "Title: Low-Carb Berry Greek Yogurt Parfait\n"
    "Ingredients:\n"
    "- plain Greek yogurt\n"
    "- stevia\n"
    "- chia seeds\n"
    "- strawberries\n"
    "- blueberries\n"
    "- chopped almonds\n"
    "Steps:\n"
    "1. Stir stevia into the Greek yogurt.\n"
    "2. Layer yogurt with berries and chia seeds in a glass.\n"
    "3. Top with chopped almonds and serve chilled.\n"
)

# === UNFILTERED TRAINING: sample 5000 recipes from RAW_recipes ===
df = pd.read_csv("data/external/foodcom/RAW_recipes.csv").sample(5000, random_state=42)

data = {"input_text": [], "target_text": []}

print("Formatting 5000 UNFILTERED recipes with diabetes context...")
for _, row in df.iterrows():
    try:
        ingredients = ast.literal_eval(row["ingredients"])
        steps = ast.literal_eval(row["steps"])
        title = str(row["name"])

        # ---- CONTEXTUAL PROMPT (unfiltered data, but rich context) ----
        prompt = (
            CONTEXT
            + "\n\n"
            + FEWSHOT_EXAMPLE
            + "\n\n"
            + f'Now generate a new, diabetes-friendly recipe called: "{title}". '
              "Return the ingredients list followed by numbered cooking steps."
        )

        # ---- TARGET: full ingredients + steps from the original recipe ----
        ingredients_text = "\n".join(f"- {ing}" for ing in ingredients)
        steps_text = "\n".join(
            f"{i+1}. {s}" for i, s in enumerate(steps)
        )

        target = f"Ingredients:\n{ingredients_text}\n\nSteps:\n{steps_text}"

        data["input_text"].append(prompt)
        data["target_text"].append(target)
    except Exception:
        # if parsing fails, skip that row
        continue

print(f"Created {len(data['input_text'])} training examples")

dataset = Dataset.from_dict(data)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# LoRA config (same as before)
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
        batch["input_text"], max_length=256, truncation=True, padding=False
    )
    labels = tokenizer(
        text_target=batch["target_text"],
        max_length=384,
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
print("UNFILTERED + CONTEXT LORA FINE-TUNING (TASK 1)")
print("=" * 60)
print(f"Model: T5-Base with LoRA (r=8)")
print(f"Training samples: {len(tokenized)}")
print("Dataset: RAW_recipes (no diabetes filter)")
print("Prompt: diabetes rules + in-context example + recipe title")
print(f"Epochs: 3")
print(f"Trainable params: {100 * trainable / total:.2f}%")
print("=" * 60 + "\n")

trainer.train()

print("\n" + "=" * 60)
print("SAVING UNFILTERED + CONTEXT MODEL")
print("=" * 60)
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print(f"✅ Unfiltered + context model saved to {MODEL_DIR}!")
