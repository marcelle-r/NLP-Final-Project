import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from tqdm import tqdm
import torch

PROMPT_TEMPLATE = "Generate a diabetes-friendly {dish} recipe."
BASE_MODEL = "google/flan-t5-base"
LORA_PATH = "models/lora_t5_base_5000_unfiltered_rl_lcr"   # <-- RL model

def generate_rl(prompts_file, output_file):
    base_dir = Path(__file__).resolve().parents[1]
    prompts_path = base_dir / "data" / prompts_file
    out_path = base_dir / "outputs" / output_file

    df = pd.read_csv(prompts_path)
    df["prompt"] = df["dish"].apply(lambda d: PROMPT_TEMPLATE.format(dish=d))

    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

    print(f"Loading RL LoRA adapters from: {LORA_PATH}")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model = model.merge_and_unload()

    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    print("=== RL MODEL ON REALISTIC TEST ===")
    generate_rl("prompts_from_safe.csv", "finetuned_rl_realistic.csv")

    print("=== RL MODEL ON ADVERSARIAL TEST ===")
    generate_rl("prompts_dev.csv", "finetuned_rl_adversarial.csv")

