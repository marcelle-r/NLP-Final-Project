#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from tqdm import tqdm
import torch

PROMPT_TEMPLATE = "Generate a diabetes-friendly {dish} recipe."
BASE_MODEL = "google/flan-t5-base"

# *** use the UNFILTERED model we just trained ***
LORA_PATH = "models/lora_t5_base_5000_unfiltered"


def generate_with_finetuned(prompts_file: str, output_file: str):
    base_dir = Path(__file__).resolve().parents[1]
    prompts_path = base_dir / "data" / prompts_file
    out_path = base_dir / "outputs" / output_file

    print(f"Loading prompts from: {prompts_path}")
    df = pd.read_csv(prompts_path)
    df["prompt"] = df["dish"].apply(lambda d: PROMPT_TEMPLATE.format(dish=d))

    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

    print(f"Loading LoRA weights from: {LORA_PATH}")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    # merge LoRA weights into the base model for faster inference
    model = model.merge_and_unload()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Using device: {device}")

    generations = []
    print(f"\nGenerating {len(df)} recipes...")
    for prompt in tqdm(df["prompt"], desc="Generating"):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=4,
            do_sample=False,
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generations.append(text)

    df["generation"] = generations
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n✅ Saved to {out_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING (UNFILTERED LORA): ADVERSARIAL TEST")
    print("=" * 60)
    generate_with_finetuned(
        "prompts_dev.csv",
        "finetuned_unfiltered_adversarial.csv",
    )

    print("\n" + "=" * 60)
    print("GENERATING (UNFILTERED LORA): REALISTIC TEST")
    print("=" * 60)
    generate_with_finetuned(
        "prompts_from_safe.csv",
        "finetuned_unfiltered_realistic.csv",
    )

    print("\n✅ Both test sets generated with UNFILTERED model!")

