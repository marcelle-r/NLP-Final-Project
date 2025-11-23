import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import torch

PROMPT_TEMPLATE = "Generate a diabetes-friendly {dish} recipe."

def load_prompts(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def build_inputs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["prompt"] = df["dish"].apply(lambda d: PROMPT_TEMPLATE.format(dish=d))
    return df

def generate_recipes(
    df: pd.DataFrame,
    model_name: str = "google/flan-t5-large",
    max_new_tokens: int = 256,
    num_beams: int = 4,
):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        model = model.to(device)
    else:
        print("WARNING: Running on CPU - this will be very slow!")

    generations = []
    print(f"\nGenerating {len(df)} recipes...")

    # Progress bar with tqdm
    for prompt in tqdm(df["prompt"], desc="Generating recipes", unit="recipe"):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generations.append(text)

    df_out = df.copy()
    df_out["generation"] = generations
    return df_out

def main():
    base_dir = Path(__file__).resolve().parents[1]
    prompts_path = base_dir / "data" / "prompts_from_safe.csv"
    out_path = base_dir / "outputs" / "baseline_realistic_generations.csv"

    print(f"Loading prompts from: {prompts_path}")
    df_prompts = load_prompts(prompts_path)
    df_inputs = build_inputs(df_prompts)

    df_gen = generate_recipes(df_inputs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_gen.to_csv(out_path, index=False)
    print(f"\n✅ Saved baseline realistic generations to {out_path}")

    # Show examples
    print("\n--- Sample Generations ---")
    for i in range(min(3, len(df_gen))):
        print(f"\n{i+1}. Dish: {df_gen.iloc[i]['dish']}")
        print(f"   Generation: {df_gen.iloc[i]['generation'][:150]}...")

if __name__ == "__main__":
    main()
