# scripts/generate_safe_full_realistic.py

#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from tqdm import tqdm

PROMPT_TEMPLATE = "Generate a diabetes-friendly {dish} recipe."
BASE_MODEL = "google/flan-t5-base"
LORA_PATH = "models/lora_t5_base_safe_full"

def generate_safe_full(prompts_file, output_file):
    base_dir = Path(__file__).resolve().parents[1]
    prompts_path = base_dir / "data" / prompts_file
    out_path = base_dir / "outputs" / output_file

    df = pd.read_csv(prompts_path)
    df["prompt"] = df["dish"].apply(lambda d: PROMPT_TEMPLATE.format(dish=d))

    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

    print(f"Loading LoRA adapters from: {LORA_PATH}")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model = model.merge_and_unload()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)

    generations = []
    print(f"Generating {len(df)} recipes...")
    for p in tqdm(df["prompt"], desc="Generating"):
        inputs = tokenizer(p, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=256, num_beams=4)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generations.append(text)

    df["generation"] = generations
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"✅ Saved to {out_path}")

if __name__ == "__main__":
    print("=== SAFE FULL MODEL ON REALISTIC TEST ===")
    generate_safe_full("prompts_from_safe.csv", "finetuned_safe_full_realistic.csv")

