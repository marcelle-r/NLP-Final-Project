from pathlib import Path
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

MODEL_NAME = "google/flan-t5-large"

def load_safe_recipes(path: Path):
    df = pd.read_csv(path)
    # you can choose a subset / specific columns
    return df

def build_dataset(df: pd.DataFrame, tokenizer):
    def format_example(row):
        title = row.get("name", "recipe")
        ingredients = row.get("ingredients_list", row.get("ingredients", ""))
        if isinstance(ingredients, str):
            ingredients_text = ingredients
        else:
            ingredients_text = ", ".join(ingredients)

        # Input: just dish type for now
        prompt = f"Generate a diabetes-friendly {title} recipe."
        # Target: we use the original text as training target
        target = f"Ingredients:\n{ingredients_text}\n\nDirections:\n<fill directions here>"
        # For now, if dataset has no directions, you can simplify later

        return {"input_text": prompt, "target_text": target}

    formatted = df.apply(format_example, axis=1, result_type="expand")
    dataset = Dataset.from_pandas(formatted)

    def tokenize_fn(batch):
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=128,
            truncation=True,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["target_text"],
                max_length=256,
                truncation=True,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

def main():
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / "processed" / "foodcom_safe_recipes.csv"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q", "v"],  # fine-tune attention projections
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora_config)

    df = load_safe_recipes(data_path)
    dataset = build_dataset(df, tokenizer)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = dataset["train"]
    eval_ds = dataset["test"]

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    output_dir = base_dir / "models" / "lora_t5"
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=1e-4,
        num_train_epochs=3,
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=False,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved LoRA-adapted model to {output_dir}")

if __name__ == "__main__":
    main()

